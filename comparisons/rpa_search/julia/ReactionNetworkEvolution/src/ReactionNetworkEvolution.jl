module ReactionNetworkEvolution
export run_evolution
export process_oscillators

import Dates: now, format

include("evo_utils.jl")
include("process_output.jl")

function evolve_networks(parentdir::String, settings::Settings, objfunct::ObjectiveFunction)

    if settings.track_fitness
        tracker = Dict{String, Any}(
            "top_individual_fitness" => Vector{Float64}(),
        )
    end
    
    ng = get_networkgenerator(settings)

    DELTA = settings.starting_delta

    population = generate_network_population(settings, ng)
    species_by_IDs = initialize_species_by_IDs(population)

    if settings.enable_speciation
        species_by_IDs, DELTA = speciate(species_by_IDs, population, DELTA, settings)
    end

    
    for i in 1:settings.ngenerations
    
        species_by_IDs, total_fitness = evaluate_population_fitness(objfunct, species_by_IDs, settings)
        species_by_IDs, total_offspring = calculate_num_offspring(species_by_IDs, total_fitness, settings, writeoutdir=joinpath(parentdir, "stalled_models"))

        bestnetwork, maxfitness = gettopmodel(species_by_IDs)

        if settings.track_fitness
            push!(tracker["top_individual_fitness"], maxfitness)
        end

        if maxfitness > settings.writeout_threshold 
            break 
        end

        population = reproduce_networks(species_by_IDs, settings, ng, objfunct, total_offspring)

        if settings.enable_speciation
            species_by_IDs, DELTA = speciate(species_by_IDs, population, DELTA, settings)
        else
            species_by_IDs = initialize_species_by_IDs(population)
        end
    end
    
    species_by_IDs, total_fitness = evaluate_population_fitness(objfunct, species_by_IDs, settings)
    bestnetwork, maxfitness = gettopmodel(species_by_IDs)

    if settings.track_fitness
        push!(tracker["top_individual_fitness"],maxfitness)
    end

    writeoutnetwork(bestnetwork, "$(bestnetwork.ID).ant", parentdir)

    if settings.track_fitness
        stringtracker = json(tracker)
        open(joinpath(parentdir, "$(bestnetwork.ID)_fitness.json"), "w") do f
            write(f, stringtracker)
        end
    end

end

function run_evolution(;
    ngenerations::Int64=-1,
    ntrials::Int64=-1,
    nspecies::Int64=3,
    initial_concentrations::Vector{Float64}=[1., 5., 9.],
    population_size::Int64=-1,
    pathtosettings::String="",
    outputpath::String="",
    seed::Int64=-1,
    note::String="")

    
    println("Starting $(format(now(), "yyyy-mm-dd HH:MM:SS"))")

    if ntrials == -1
        ntrials = 100
    end
    if outputpath == ""
        outputpath = joinpath(pwd(), "evolution_output")
    end
    # Create a directory of output of this batch in the data dir
    if !isdir(outputpath)
        # This is for dealing with race conditions in parallel computing
        try
            mkdir(outputpath)
        catch LoadError
        end
    end

    println("Writing output to $outputpath")
    starttime = format(now(), "yyyy-mm-dd_HHMMSS")
    path = joinpath(outputpath, "batch_$starttime" * randstring(3)) # randstring is for race conditions

    if !isdir(path)
        mkdir(path)
    end

    # If no path to settings is supplied, use default settings
    if pathtosettings == ""
        pathtosettings = "DEFAULT"
    end

    settings, objectivefunction = read_usersettings(pathtosettings, ngenerations=ngenerations, nspecies=nspecies, initial_concentrations=initial_concentrations, population_size=population_size, seed=seed, note=note)

    println("Beginning $ntrials trials with $(settings.ngenerations) generations each")

    @time for i in 1:ntrials
        if settings.verbose && (i%10 == 0 || i == ntrials)
            println(". trial $i")
        elseif settings.verbose
            print(".")
        end
        evolve_networks(path, settings, objectivefunction)
    end

    writeout_settings(settings, joinpath(outputpath, "settings.json"))

    if settings.process_output_oscillators
        println("Sorting oscillators...")
        process_oscillators(path, track_fitness=settings.track_fitness)
    else
        println("Output written to $path")
    end

    println("Done")
end

end #module

