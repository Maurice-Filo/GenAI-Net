using Dates
using DifferentialEquations
using JSON
using Random
using Sundials

include(joinpath(@__DIR__, "ReactionNetworkEvolution", "src", "evo_utils.jl"))

mutable struct RpaEvalContext
    config::Dict{String, Any}
    run_dir::String
    method::String
    run_id::String
    candidate_budget::Int
    candidate_evaluations::Int
    ode_simulations::Int
    scenario_evaluations::Int
    best_loss::Float64
    best_network::Any
    tic::Float64
end

const CURRENT_CTX = Ref{Union{Nothing, RpaEvalContext}}(nothing)

function append_line(path::String, line::String)
    mkpath(dirname(path))
    open(path, "a") do io
        write(io, line * "\n")
    end
end

function ensure_csv_headers(run_dir::String)
    progress = joinpath(run_dir, "progress.csv")
    if !isfile(progress)
        append_line(progress, "method,run_id,step,candidate_evaluations,ode_simulations,scenario_count,scenario_evaluations,loss,best_so_far_loss,performance,best_so_far_performance,elapsed_seconds")
    end
    candidates = joinpath(run_dir, "candidates.csv")
    if !isfile(candidates)
        append_line(candidates, "method,run_id,candidate_id,candidate_evaluations,ode_simulations,scenario_count,scenario_evaluations,loss,best_so_far_loss,reaction_ids,rate_constants")
    end
end

function reaction_strings(network)
    out = String[]
    rates = Float64[]
    for key in keys(network.reactionlist)
        r = network.reactionlist[key]
        if r.isactive
            push!(out, "$(join(r.substrate, "+"))->$(join(r.product, "+"))")
            push!(rates, r.rateconstant)
        end
    end
    return out, rates
end

function valid_state(x, large_number::Float64)
    max_val = maximum(abs.(x))
    return isfinite(max_val) && max_val < large_number
end

function write_best_network(ctx::RpaEvalContext)
    ctx.best_network === nothing && return
    rxns, rates = reaction_strings(ctx.best_network)
    open(joinpath(ctx.run_dir, "best_network.txt"), "w") do io
        println(io, "fitness: ", ctx.best_network.fitness)
        println(io, "loss: ", ctx.best_loss)
        for (rxn, rate) in zip(rxns, rates)
            println(io, rxn, " ; k=", rate)
        end
    end
    data = Dict(
        "loss" => ctx.best_loss,
        "fitness" => ctx.best_network.fitness,
        "reactions" => rxns,
        "rate_constants" => rates,
    )
    open(joinpath(ctx.run_dir, "best_network.json"), "w") do io
        JSON.print(io, data, 2)
    end
end

function rpa_rhs!(du, x, p, t)
    network, u1, u2 = p
    du[1] = u1
    du[2] = 0.0
    du[3] = -u2 * x[3]

    species = network.chemical_species_names
    for key in keys(network.reactionlist)
        r = network.reactionlist[key]
        r.isactive || continue
        rate = r.rateconstant
        for s in r.substrate
            idx = findfirst(==(s), species)
            rate *= x[idx]
        end
        for s in r.substrate
            idx = findfirst(==(s), species)
            du[idx] -= rate
        end
        for s in r.product
            idx = findfirst(==(s), species)
            du[idx] += rate
        end
    end
    return nothing
end

function rpa_loss(network, ctx::RpaEvalContext)
    rpa = ctx.config["rpa"]
    rne = get(ctx.config, "rne", Dict{String, Any}())
    large_number = Float64(get(rne, "LARGE_NUMBER", 1.0e4))
    t_f = Float64(rpa["t_f"])
    n_t = Int(rpa["n_t"])
    times = collect(range(0.0, t_f, length=n_t))
    u_values = Float64.(rpa["u_values"])
    x0 = fill(Float64(rpa["ic"][2]), 3)

    weights = ones(Float64, n_t)
    fifth = div(n_t, 5)
    weights[1:fifth] .*= 0.25
    weights[(4 * fifth + 1):end] .*= 2.0

    total = 0.0
    n_terms = 0
    for u1 in u_values, u2 in u_values
        prob = ODEProblem(rpa_rhs!, x0, (0.0, t_f), (network, u1, u2))
        sol = solve(prob, CVODE_BDF(), saveat=times, reltol=Float64(rpa["rtol"]), abstol=Float64(rpa["atol"]), verbose=false)
        if sol.retcode != ReturnCode.Success || length(sol.u) != n_t
            return large_number
        end
        for i in 1:n_t
            if !valid_state(sol.u[i], large_number)
                return large_number
            end
            y = sol.u[i][3]
            total += weights[i] * abs(u1 - y)
            n_terms += 1
        end
    end
    return total / n_terms
end

function logic_rhs!(du, x, p, t)
    network, u = p
    n_inputs = length(u)
    for i in eachindex(du)
        du[i] = 0.0
    end
    for i in 1:n_inputs
        du[i] = u[i]
    end

    species = network.chemical_species_names
    for key in keys(network.reactionlist)
        r = network.reactionlist[key]
        r.isactive || continue
        rate = r.rateconstant
        for s in r.substrate
            idx = findfirst(==(s), species)
            rate *= x[idx]
        end
        for s in r.substrate
            idx = findfirst(==(s), species)
            du[idx] -= rate
        end
        for s in r.product
            idx = findfirst(==(s), species)
            du[idx] += rate
        end
    end
    return nothing
end

function logic_target(u)
    return ((u[1] > 0.5 && u[2] > 0.5) ||
            (u[2] > 0.5 && u[3] > 0.5) ||
            (u[3] > 0.5 && u[4] > 0.5)) ? 1.0 : 0.0
end

function logic_loss(network, ctx::RpaEvalContext)
    logic = ctx.config["logic"]
    rne = get(ctx.config, "rne", Dict{String, Any}())
    large_number = Float64(get(rne, "LARGE_NUMBER", 1.0e4))
    t_f = Float64(logic["t_f"])
    n_t = Int(logic["n_t"])
    n_inputs = Int(get(logic, "n_inputs", 4))
    n_species = length(network.chemical_species_names)
    times = collect(range(0.0, t_f, length=n_t))
    x0 = fill(Float64(logic["ic"][2]), n_species)

    weights = ones(Float64, n_t)
    fifth = div(n_t, 5)
    weights[1:fifth] .*= 0.25
    weights[(4 * fifth + 1):end] .*= 2.0

    total = 0.0
    n_terms = 0
    for mask in 0:(2^n_inputs - 1)
        u = [Float64((mask >> (i - 1)) & 1) for i in 1:n_inputs]
        target = logic_target(u)
        prob = ODEProblem(logic_rhs!, x0, (0.0, t_f), (network, u))
        sol = solve(prob, CVODE_BDF(), saveat=times, reltol=Float64(logic["rtol"]), abstol=Float64(logic["atol"]), verbose=false)
        if sol.retcode != ReturnCode.Success || length(sol.u) != n_t
            return large_number
        end
        for i in 1:n_t
            if !valid_state(sol.u[i], large_number)
                return large_number
            end
            y = sol.u[i][n_species]
            total += weights[i] * abs(target - y)
            n_terms += 1
        end
    end
    return total / n_terms
end

function task_loss(network, ctx::RpaEvalContext)
    task = get(ctx.config, "task", get(ctx.config["benchmark"], "task", "rpa"))
    if task == "logic"
        return logic_loss(network, ctx)
    end
    return rpa_loss(network, ctx)
end

function scenario_count(ctx::RpaEvalContext)
    task = get(ctx.config, "task", get(ctx.config["benchmark"], "task", "rpa"))
    if task == "logic"
        return 2 ^ Int(get(ctx.config["logic"], "n_inputs", 4))
    end
    return length(ctx.config["rpa"]["u_values"]) ^ 2
end

function evaluate_and_log(network, settings, ctx::RpaEvalContext)
    if ctx.candidate_evaluations >= ctx.candidate_budget
        return network.fitness
    end

    repair_active_reaction_cap!(network)
    loss = task_loss(network, ctx)
    fitness = 1.0 / max(loss, 1.0e-12)
    network.fitness = fitness

    ctx.candidate_evaluations += 1
    ctx.ode_simulations += 1
    scenarios = scenario_count(ctx)
    ctx.scenario_evaluations += scenarios
    if loss < ctx.best_loss
        ctx.best_loss = loss
        ctx.best_network = deepcopy(network)
    end

    elapsed = time() - ctx.tic
    append_line(
        joinpath(ctx.run_dir, "progress.csv"),
        join([
            ctx.method,
            ctx.run_id,
            string(ctx.candidate_evaluations),
            string(ctx.candidate_evaluations),
            string(ctx.ode_simulations),
            string(scenarios),
            string(ctx.scenario_evaluations),
            string(loss),
            string(ctx.best_loss),
            string(-loss),
            string(-ctx.best_loss),
            string(elapsed),
        ], ","),
    )

    rxns, rates = reaction_strings(network)
    append_line(
        joinpath(ctx.run_dir, "candidates.csv"),
        join([
            ctx.method,
            ctx.run_id,
            string(ctx.candidate_evaluations),
            string(ctx.candidate_evaluations),
            string(ctx.ode_simulations),
            string(scenarios),
            string(ctx.scenario_evaluations),
            string(loss),
            string(ctx.best_loss),
            "\"" * replace(JSON.json(rxns), "\"" => "\"\"") * "\"",
            "\"" * replace(JSON.json(rates), "\"" => "\"\"") * "\"",
        ], ","),
    )

    if ctx.candidate_evaluations == 1 || ctx.candidate_evaluations == ctx.candidate_budget || ctx.candidate_evaluations % 50 == 0
        println("[$(ctx.method) $(ctx.run_id)] candidates=$(ctx.candidate_evaluations) ode_sims=$(ctx.ode_simulations) best_loss=$(ctx.best_loss)")
        flush(stdout)
    end
    return fitness
end

function evaluate_fitness(objfunct::ObjectiveFunction, network::ReactionNetwork, settings::Settings)
    ctx = CURRENT_CTX[]
    ctx === nothing && return network.fitness
    return evaluate_and_log(network, settings, ctx)
end

function evaluate_species!(species_by_IDs, settings, ctx::RpaEvalContext)
    total_fitness = 0.0
    for speciesID in keys(species_by_IDs)
        species = species_by_IDs[speciesID]
        species_fitness = 0.0
        topfitness = -Inf
        topnetwork = species.networks[1]
        for network in species.networks
            ctx.candidate_evaluations >= ctx.candidate_budget && break
            fitness = evaluate_and_log(network, settings, ctx)
            species_fitness += fitness
            if fitness >= topfitness
                topfitness = fitness
                topnetwork = network
            end
        end
        n = max(1, length(species.networks))
        if settings.average_fitness
            total_fitness += species_fitness / n
            species.speciesfitness = species_fitness / n
        else
            total_fitness += topfitness
            species.speciesfitness = topfitness
        end
        oldfitness = species.topfitness
        species.topfitness = topfitness
        species.topnetwork = topnetwork
        species.numstagnations = oldfitness == topfitness ? species.numstagnations + 1 : 0
    end
    return species_by_IDs, total_fitness
end

function main()
    config_path = ARGS[1]
    run_dir = ARGS[2]
    run_id = ARGS[3]
    config = JSON.parsefile(config_path)
    rne_config = get(config, "rne", Dict{String, Any}())
    constrain_reactions = Bool(get(rne_config, "constrain_reactions", false))
    bounded_state = Bool(get(rne_config, "bounded_state", false))
    method = constrain_reactions && bounded_state ? "reaction_network_evolution_jl_constrained_bounded" :
             constrain_reactions ? "reaction_network_evolution_jl_constrained" :
             "reaction_network_evolution_jl"
    search = config["search"]

    mkpath(run_dir)
    cp(config_path, joinpath(run_dir, "config.json"); force=true)
    ensure_csv_headers(run_dir)

    seed = Int(search["seed"])
    Random.seed!(seed)
    population_size = Int(get(rne_config, "population_size", 64))
    candidate_budget = Int(search["candidate_budget"])
    max_generations = Int(ceil(candidate_budget / population_size)) + 2
    max_added_reactions = Int(search["max_added_reactions"])
    set_active_reaction_cap!(constrain_reactions ? max_added_reactions : nothing)

    task = get(config, "task", get(config["benchmark"], "task", "rpa"))
    chemical_species_names = task == "logic" ?
        vcat(["X_$(i)" for i in 1:Int(get(config["logic"], "n_inputs", 4))], ["OUT"]) :
        config["rpa"]["species"]
    initial_concentrations = task == "logic" ?
        fill(Float64(config["logic"]["ic"][2]), length(chemical_species_names)) :
        fill(Float64(config["rpa"]["ic"][2]), length(chemical_species_names))

    settings = read_usersettings(Dict{String, Any}(
        "chemical_species_names" => chemical_species_names,
        "initial_concentrations" => initial_concentrations,
        "rateconstant_range" => Float64.(search["rate_constant_range"]),
        "population_size" => population_size,
        "ngenerations" => max_generations,
        "nreactions" => max_added_reactions,
        "reaction_probabilities" => [0.25, 0.25, 0.25, 0.25],
        "process_output_oscillators" => false,
        "verbose" => false,
        "seed" => seed,
    ))

    ctx = RpaEvalContext(config, run_dir, method, run_id, candidate_budget, 0, 0, 0, Inf, nothing, time())
    CURRENT_CTX[] = ctx

    ng = get_networkgenerator(settings)
    population = generate_network_population(settings, ng)
    if constrain_reactions
        foreach(repair_active_reaction_cap!, population)
    end
    species_by_IDs = initialize_species_by_IDs(population)
    delta = settings.starting_delta

    if settings.enable_speciation
        species_by_IDs, delta = speciate(species_by_IDs, population, delta, settings)
    end

    while ctx.candidate_evaluations < ctx.candidate_budget
        species_by_IDs, total_fitness = evaluate_species!(species_by_IDs, settings, ctx)
        ctx.candidate_evaluations >= ctx.candidate_budget && break

        species_by_IDs, total_offspring = calculate_num_offspring(species_by_IDs, max(total_fitness, 1.0e-12), settings)
        population = reproduce_networks(species_by_IDs, settings, ng, ObjectiveFunction([], Float64[]), total_offspring)
        if constrain_reactions
            foreach(repair_active_reaction_cap!, population)
        end

        if settings.enable_speciation
            species_by_IDs, delta = speciate(species_by_IDs, population, delta, settings)
        else
            species_by_IDs = initialize_species_by_IDs(population)
        end
    end

    write_best_network(ctx)
end

main()
