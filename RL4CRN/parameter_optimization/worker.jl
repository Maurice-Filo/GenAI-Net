# Private JSON-lines worker. Reuses installed SynthEvo dependencies without
# loading Catalyst/Plots or mutating the sibling package environment.
using LinearAlgebra, SHA
BLAS.set_num_threads(1)
const JSON = Base.require(Base.PkgId(Base.UUID("682c06a0-de6a-54ab-a142-c8b1cf79cde6"), "JSON"))
const SciML = Base.require(Base.PkgId(Base.UUID("0bca4576-84f4-4d90-8ffe-ffa030f20462"), "SciMLBase"))
const Sundials = Base.require(Base.PkgId(Base.UUID("c3572dad-4567-51f8-b174-8c6c989267f4"), "Sundials"))
const Tsit = Base.require(Base.PkgId(Base.UUID("b1df2697-797e-41e3-8120-5422d3b24e4a"), "OrdinaryDiffEqTsit5"))
const optimizer_source = joinpath(dirname(Base.active_project()), "src", "CRNOptim.jl")
include(optimizer_source)

struct CompiledReaction
    reactants::Vector{Int}
    affected::Vector{Int}
    changes::Vector{Int}
    input::Int
    parameter::Int
    fixed_rate::Float64
end

struct Network
    n::Int
    p::Int
    reactions::Vector{CompiledReaction}
end

struct EvaluationFailure <: Exception
    message::String
end
Base.showerror(io::IO, e::EvaluationFailure) = print(io, e.message)

function rhs!(dz, z, parameters, t)
    network, rates, inputs = parameters
    n, count = network.n, network.p
    fill!(dz, 0.)
    # Forward sensitivities: S' = (df/dx) S + df/dk. Products excluding
    # one reactant avoid division by zero and preserve repeated-species powers.
    for r in network.reactions
        modulation = r.input == 0 ? 1. : inputs[r.input]
        base = modulation
        for a in r.reactants
            base *= z[a]
        end
        k = r.parameter == 0 ? r.fixed_rate : rates[r.parameter]
        for a in eachindex(r.affected)
            dz[r.affected[a]] += r.changes[a] * k * base
        end
        for q in 1:count
            derivative = r.parameter == q ? base : 0.
            for j in eachindex(r.reactants)
                partial = modulation
                for l in eachindex(r.reactants)
                    if l != j
                        partial *= z[r.reactants[l]]
                    end
                end
                derivative += k * partial * z[n + (q-1)*n + r.reactants[j]]
            end
            for a in eachindex(r.affected)
                dz[n + (q-1)*n + r.affected[a]] += r.changes[a] * derivative
            end
        end
    end
    return nothing
end

matrix(rows) = reduce(vcat, [permutedims(Float64.(row)) for row in rows])

struct Prepared
    network::Network
    times::Vector{Float64}
    output::Matrix{Float64}
    targets::Vector{Matrix{Float64}}
    weights::Vector{Matrix{Float64}}
    problems::Vector{SciML.ODEProblem}
    loss::String
    relative::Bool
    solver::String
    rtol::Float64
    atol::Float64
    max_steps::Int
    max_species::Float64
end

function prepare(request)
    crn, ex = request["crn"], request["experiment"]
    n, count = length(crn["species"]), 0
    reactions = CompiledReaction[]
    for r in crn["reactions"]
        free = r["fixed_rate"] === nothing
        count += free
        reactants = [i for i in 1:n for _ in 1:Int(r["reactants"][i])]
        net = Int.(r["products"]) - Int.(r["reactants"])
        affected = findall(!iszero, net)
        push!(reactions, CompiledReaction(reactants, affected, net[affected],
              r["input_channel"] === nothing ? 0 : Int(r["input_channel"]) + 1,
              free ? count : 0, free ? 0. : Float64(r["fixed_rate"])))
    end
    network = Network(n, count, reactions)
    times = Float64.(ex["times"])
    problems = SciML.ODEProblem[]
    for (x0, input) in zip(ex["initial_states"], ex["inputs"])
        z0 = [Float64.(x0); zeros(n*count)]
        p = (network, ones(count), Float64.(input))
        push!(problems, SciML.ODEProblem(rhs!, z0, (first(times), last(times)), p))
    end
    return Prepared(network, times, matrix(ex["output_matrix"]),
        matrix.(ex["targets"]), matrix.(ex["weights"]), problems, ex["loss"],
        ex["relative"], ex["solver"], ex["rtol"], ex["atol"], ex["max_steps"],
        ex["max_species"] === nothing ? Inf : ex["max_species"])
end

function evaluate(model, rates; trajectories=false, solve_counter=Ref(0))
    n, count = model.network.n, model.network.p
    length(rates) == count && all(isfinite, rates) && all(>(0), rates) || error("Invalid rates")
    grad, loss, peak_species = zeros(count), 0., 0.
    outputs = Matrix{Float64}[]
    algorithm = model.solver == "cvode" ? Sundials.CVODE_BDF() : Tsit.Tsit5()
    callback = if isfinite(model.max_species)
        SciML.ContinuousCallback(
            (z, t, integrator) -> model.max_species - maximum(abs, view(z, 1:n)),
            integrator -> SciML.terminate!(integrator); save_positions=(false, false))
    else
        nothing
    end
    norm = length(model.problems) * size(model.output, 1) * length(model.times)
    solves = 0
    for (scenario, base) in enumerate(model.problems)
        problem = SciML.remake(base; p=(model.network, rates, base.p[3]))
        solve_counter[] += 1
        sol = SciML.solve(problem, algorithm; reltol=model.rtol, abstol=model.atol,
            saveat=model.times, save_everystep=false, dense=false, maxiters=model.max_steps,
            callback=callback)
        solves += 1
        if sol.retcode == SciML.ReturnCode.Terminated
            throw(EvaluationFailure("Species ceiling $(model.max_species) reached; trajectory is inadmissible"))
        end
        if !SciML.successful_retcode(sol) || length(sol.t) != length(model.times) || sol.t != model.times
            throw(EvaluationFailure("Incomplete integration: $(sol.retcode), saved $(length(sol.t))/$(length(model.times)) times"))
        end
        trajectory = trajectories ? zeros(size(model.output, 1), length(model.times)) : nothing
        for (it, z) in enumerate(sol.u)
            peak_species = max(peak_species, maximum(view(z, 1:n)))
            if !all(isfinite, z) || minimum(view(z, 1:n)) < -10model.atol
                throw(EvaluationFailure("Nonfinite state/sensitivity or negative species beyond numerical tolerance"))
            end
            if maximum(abs, view(z, 1:n)) >= model.max_species
                throw(EvaluationFailure("Species ceiling reached at a saved state"))
            end
            for o in axes(model.output, 1)
                y = dot(view(model.output, o, :), view(z, 1:n))
                trajectories && (trajectory[o, it] = y)
                target = model.targets[scenario][o, it]
                scale = model.relative ? max(abs(target), 1e-6) : 1.
                err = (y - target) / scale
                weight = model.weights[scenario][o, it] / norm
                loss += weight * (model.loss == "l1" ? abs(err) : err^2)
                slope = weight * (model.loss == "l1" ? sign(err) : 2err) / scale
                for q in 1:count
                    grad[q] += slope * dot(view(model.output, o, :), view(z, n*q+1:n*(q+1)))
                end
            end
        end
        trajectories && push!(outputs, trajectory)
    end
    isfinite(loss) && all(isfinite, grad) || throw(EvaluationFailure("Nonfinite loss/gradient"))
    result = Dict{String,Any}("loss"=>loss, "gradient"=>grad, "scenario_solves"=>solves,
                              "peak_species"=>peak_species)
    trajectories && (result["outputs"] = [[collect(row) for row in eachrow(y)] for y in outputs])
    return result
end

function optimize_one(model, key, start, checkpoint, steps, config)
    count = model.network.p
    lo, hi = Float64.(config["bounds"])
    lr = Float64(config["learning_rate"])
    method = config["method"]
    state = checkpoint
    initial_clipped = false
    if state === nothing
        if length(start) != count || any(x -> x === nothing || !isfinite(x) || x <= 0, start)
            return Dict("status"=>"invalid_start", "error"=>"Expected positive finite start rates")
        end
        initial = Float64.(start)
        initial_clipped = any(x -> x < lo || x > hi, initial)
        state = Dict{String,Any}("key"=>key, "config"=>config, "step"=>0,
            "theta"=>log.(clamp.(initial, lo, hi)), "accumulator"=>zeros(count),
            "moment"=>zeros(count), "velocity"=>zeros(count), "best_loss"=>nothing,
            "best_rates"=>nothing, "current"=>nothing)
    else
        state = deepcopy(state)
        state["key"] == key && state["config"] == config || error("Incompatible checkpoint")
    end
    for name in ("theta", "accumulator", "moment", "velocity")
        state[name] = Float64.(state[name])
        length(state[name]) == count && all(isfinite, state[name]) || error("Invalid checkpoint $name")
    end
    history = Any[]
    evaluations, completed_evaluations, solve_counter = 0, 0, Ref(0)
    rates = clamp.(exp.(state["theta"]), lo, hi)
    started = time_ns()
    status, failure = "ok", nothing
    current = state["current"]
    try
        for iteration in 0:steps
            if current === nothing
                evaluations += 1
                current = evaluate(model, rates; solve_counter=solve_counter)
                completed_evaluations += 1
            end
            if state["best_loss"] === nothing || current["loss"] < state["best_loss"]
                state["best_loss"], state["best_rates"] = current["loss"], copy(rates)
            end
            push!(history, Dict("step"=>state["step"], "loss"=>current["loss"],
                               "best_loss"=>state["best_loss"]))
            state["current"] = current
            iteration == steps && break
            gradient = rates .* Float64.(current["gradient"])
            state["step"] += 1
            delta = if method == "adagrad"
                coefficient = adagrad_update_get_coefficient(rates, gradient, state["accumulator"], lr)
                coefficient .* gradient
            else
                ADAM_update_get_coefficient(rates, gradient, state["moment"], state["velocity"],
                                           lr, state["step"], .9, .999, 1e-8)
            end
            all(isfinite, delta) || error("Nonfinite optimizer step")
            state["theta"] = clamp.(state["theta"] - delta, log(lo), log(hi))
            rates = clamp.(exp.(state["theta"]), lo, hi)
            current = nothing
            state["current"] = nothing
        end
    catch error_value
        error_value isa EvaluationFailure || rethrow()
        status, failure = "evaluation_failed", sprint(showerror, error_value)
        # The last successful best remains explicitly distinct from a failed final iterate.
        current = nothing
        state["current"] = nothing
    end
    return Dict("status"=>status, "error"=>failure, "best_loss"=>state["best_loss"],
        "best_rates"=>state["best_rates"], "final_rates"=>rates,
        "final_loss"=>current === nothing ? nothing : current["loss"],
        "state"=>state, "history"=>history, "evaluations"=>evaluations,
        "completed_evaluations"=>completed_evaluations,
        "scenario_solves"=>solve_counter[], "initial_clipped"=>initial_clipped,
        "elapsed_s"=>(time_ns()-started)/1e9)
end

const feasible_source = joinpath(@__DIR__, "feasible.jl")
include(feasible_source)

const models = Dict{String,Prepared}()
const prepares = Ref(0)
function dispatch(request)
    command = request["command"]
    if command == "stats"
        return Dict("pid"=>getpid(), "models"=>length(models), "prepares"=>prepares[])
    elseif command == "prepare"
        started = time_ns()
        models[request["key"]] = prepare(request)
        prepares[] += 1
        return Dict("key"=>request["key"], "prepare_s"=>(time_ns()-started)/1e9)
    elseif command == "release"
        delete!(models, request["key"])
        return Dict("status"=>"ok")
    end
    key = request["key"]
    model = models[key]
    if command == "evaluate"
        return evaluate(model, Float64.(request["rates"]); trajectories=request["trajectories"])
    elseif command == "optimize"
        checkpoints = request["states"]
        starts = request["starts"]
        count = checkpoints === nothing ? length(starts) : length(checkpoints)
        results = Any[]
        for i in 1:count
            optimizer = get(request["config"], "step_policy", "stop") == "feasible" ? optimize_feasible : optimize_one
            args = (model, key, starts === nothing ? nothing : starts[i],
                checkpoints === nothing ? nothing : checkpoints[i], request["steps"], request["config"])
            push!(results, optimizer === optimize_feasible ?
                optimizer(args...; evaluation_limit=get(request, "evaluation_limit", nothing)) : optimizer(args...))
        end
        return Dict("results"=>results, "pid"=>getpid())
    end
    error("Unsupported command: $command")
end

function send(value)
    println(JSON.json(value))
    flush(stdout)
end

send(Dict("status"=>"ready", "pid"=>getpid(), "julia"=>string(VERSION),
    "sundials"=>string(pkgversion(Sundials)), "scimlbase"=>string(pkgversion(SciML)),
    "manifest_sha256"=>bytes2hex(sha256(read(joinpath(dirname(Base.active_project()), "Manifest.toml")))),
    "optimizer_source_sha256"=>bytes2hex(sha256(read(optimizer_source))),
    "feasible_source_sha256"=>bytes2hex(sha256(read(feasible_source))),
    "worker_sha256"=>bytes2hex(sha256(read(@__FILE__))), "threads"=>Threads.nthreads()))
for line in eachline(stdin)
    try
        request = JSON.parse(line)
        # Dependency diagnostics must not contaminate the JSON protocol stream.
        result = redirect_stdout(stderr) do
            dispatch(request)
        end
        send(result)
    catch error_value
        send(Dict("status"=>"error", "error"=>sprint(showerror, error_value)))
    end
end
