# Read-only instrumentation of the production worker, in a separate process.
include(joinpath(@__DIR__, "../../RL4CRN/parameter_optimization/worker.jl"))
using Profile

function solver_only(model, rates)
    n = model.network.n
    callback = SciML.ContinuousCallback(
        (z, t, integrator) -> model.max_species - maximum(abs, view(z, 1:n)),
        integrator -> SciML.terminate!(integrator); save_positions=(false, false))
    last = nothing
    for base in model.problems
        problem = SciML.remake(base; p=(model.network, rates, base.p[3]))
        last = SciML.solve(problem, Sundials.CVODE_BDF(); reltol=model.rtol, abstol=model.atol,
            saveat=model.times, save_everystep=false, dense=false, maxiters=model.max_steps,
            callback=callback)
        SciML.successful_retcode(last) && last.t == model.times || error("Incomplete profile solve")
    end
    return last
end

function timings(f, repeats=20)
    f()
    rows = Any[]
    for _ in 1:repeats
        measurement = @timed f()
        push!(rows, Dict("seconds"=>measurement.time, "allocated_bytes"=>measurement.bytes,
                         "gc_seconds"=>measurement.gctime))
    end
    return rows
end

function main()
    fixture = JSON.parse(read(ARGS[1], String))
    model = prepare(fixture)
    rates = Float64.(fixture["rates"])
    complete = timings(() -> evaluate(model, rates))
    solves = timings(() -> solver_only(model, rates))
    z = copy(model.problems[1].u0)
    dz = similar(z)
    parameters = (model.network, rates, model.problems[1].p[3])
    rhs!(dz, z, parameters, 0.)
    rhs = timings() do
        for _ in 1:10000
            rhs!(dz, z, parameters, 0.)
        end
    end
    Profile.clear()
    Profile.@profile for _ in 1:100
        evaluate(model, rates)
    end
    io = IOBuffer()
    Profile.print(io; format=:flat, sortedby=:count, mincount=10)
    send(Dict("full_evaluation"=>complete, "solver_only"=>solves,
        "rhs_10000"=>rhs, "profile"=>String(take!(io)), "loss"=>evaluate(model, rates)["loss"]))
end
main()
