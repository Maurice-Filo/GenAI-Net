using Random
using JSON: json, parsefile
using DataFrames: DataFrame
using CSV: File

struct ReactionProbabilities
    uniuni::Float64
    unibi::Float64
    biuni::Float64
    bibi::Float64

    function ReactionProbabilities(p::Vector{Any})
        new(p[1], p[2], p[3], p[4])
    end

    function ReactionProbabilities(p::Vector{Float64})
        new(p[1], p[2], p[3], p[4])
    end

    function ReactionProbabilities(dict::Dict{String, Any})
        new(dict["uniuni"], dict["unibi"], dict["biuni"], dict["bibi"])
    end

    function ReactionProbabilities(p1::Float64, p2::Float64, p3::Float64, p4::Float64)
        new(p1, p2, p3, p4)
    end
end

struct Settings
    portion_elite::Float64
    reaction_probabilities::ReactionProbabilities
    p_rateconstant_mutation::Float64
    rateconstant_range::Vector{Float64}
    percent_rateconstant_change::Float64 # Uniform sampling across this range to change rate constant
    p_new_rateconstant::Float64 # Probability of picking new rate constant during mutation vs slight change
    population_size::Int
    ngenerations::Int
    nspecies::Int
    nreactions::Int
    max_offspring_portion::Float64 # The maximum portion of offspring a single species can have
    writeout_threshold::Float64 # Networks with this fitness or better will be saved
    p_crossover::Float64 # Probability of p_crossover
    p_mutation::Float64 # Probability of mutation (does not need to sum to 1 with p_crossover)
    excluisve_crossover_mutation::Bool # If true, either crossover OR mutation, never both
    portion_delete::Float64 # Portion of worst networks to drop in each species
    seed::Float64
    starting_delta::Float64
    delta_step::Float64
    rateconstant_distance_weight::Float64
    target_num_species::Int64
    use_seed_network::Bool # Start with a seed network
    seed_network_path::String
    randomize_seed_network_rates::Bool
    tournament_select::Bool
    chemical_species_names::Vector{String} # TODO: Maybe they don't need to define this if it's in the data?
    initial_concentrations::Vector{Float64}
    objective_data_path::String
    match_objectivefunction_species::Bool
    enable_speciation::Bool
    track_fitness::Bool
    #objectivespecies::Vector{String}
    average_fitness::Bool
    same_fitness_crossover::Bool
    same_fitness_percent_range::Float64
    lenient_crossover::Bool
    process_output_oscillators::Bool
    verbose::Bool
    note::String # User can add any description or other labels here
   
end

struct ObjectiveFunction
    #objectivespecies::Vector{String} # Can be more than 1
    objectivedata:: Any# DataFrame #TODO: Change this back?
    time::Vector{Float64}
    #indexbyspecies::Dict
end


function rowtovec(df::DataFrame, row::Int)
    # Quickly convert a row of a dataframe into a vector
    return [df[row,i] for i in 1:ncol(df)]
end

function get_objectivefunction(path::String)
    # If the user supplies a path for the objective timeseries, read it and get the chemical names from it
    if path != "DEFAULT"
        objectivedataframe = DataFrame(File(path))
        time = objectivedataframe[!, 1]
        objectivedata = objectivedataframe[!, 2:end]
        chemical_species_names = names(objectivedata)
        initial_concentrations = rowtovec(objectivedata, 1)
        objectivedata = Matrix(objectivedata)
    else
    # If no path is supplied, use the default oscillator time points
        time = collect(range(0, 1.25, length=11))
        objectivedata = [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0]
        chemical_species_names = [""]
        initial_concentrations = [0.]
    end
    return ObjectiveFunction(objectivedata, time), chemical_species_names, initial_concentrations
end

function read_usersettings(settings_dict::Dict{String, Any})
    # This version of the function takes a dictionary. It is mostly for testing
    settings = Dict(
        "portion_elite" => 0.1,
        "reaction_probabilities" => [.1, .4, .4, .1],
        "p_rateconstant_mutation" => .6,                         # Probability of changning rate constant vs reaction
        "rateconstant_range" => [0.1, 50.0],
        "percent_rateconstant_change" => 20,
        "p_new_rateconstant" => 0.15,
        "population_size" => 100,
        "ngenerations" => 800,
        "nspecies" => 3,
        "nreactions" => 5,
        "max_offspring_portion" => 0.1,
        "writeout_threshold" => 0.05,
        "p_crossover" => 0,
        "p_mutation" => 1,
        "exclusive_crossover_mutation" => false,
        "portion_delete" => 0.1,
        "seed" => -1,
        "starting_delta" => 0.65,
        "delta_step" => 0.1,
        "rateconstant_distance_weight" => 0.0,
        "target_num_species" => 10,
        "use_seed_network" => false,
        "seed_network_path" => "",
        "randomize_seed_network_rates" => true,
        "tournament_select" => false,
        "chemical_species_names" => ["S0", "S1", "S2"],
        "initial_concentrations" => [1.0, 5.0, 9.0],
        "objective_data_path" => "DEFAULT",
        "match_objectivefunction_species" => true,
        "enable_speciation" => true,
        "track_fitness" => true,
        "average_fitness" => false,
        "same_fitness_crossover" => false,
        "same_fitness_percent_range" => 5,
        "lenient_crossover" => false,
        "process_output_oscillators" => false,
        "verbose" => true,
        "note"=>""
        )

    # Set values to those supplied in settings_dict, otherwise set to default
    for k in keys(settings_dict)
        if k in keys(settings)
            settings[k] = settings_dict[k]
        elseif k ∉ keys(settings)
            error("$k not found in settings")
        end
    end

    reaction_probabilities = ReactionProbabilities(settings["reaction_probabilities"])

    usersettings = Settings(settings["portion_elite"], 
                reaction_probabilities,
                settings["p_rateconstant_mutation"],
                settings["rateconstant_range"],
                settings["percent_rateconstant_change"],
                settings["p_new_rateconstant"],
                settings["population_size"],
                settings["ngenerations"],
                settings["nspecies"],
                settings["nreactions"],
                settings["max_offspring_portion"],
                settings["writeout_threshold"],
                settings["p_crossover"],
                settings["p_mutation"],
                settings["exclusive_crossover_mutation"],
                settings["portion_delete"],
                settings["seed"],
                settings["starting_delta"],
                settings["delta_step"],
                settings["rateconstant_distance_weight"],
                settings["target_num_species"],
                settings["use_seed_network"],
                settings["seed_network_path"],
                settings["randomize_seed_network_rates"],
                settings["tournament_select"],
                settings["chemical_species_names"],
                settings["initial_concentrations"],
                settings["objective_data_path"],
                settings["match_objectivefunction_species"],
                settings["enable_speciation"],
                settings["track_fitness"],
                settings["average_fitness"],
                settings["same_fitness_crossover"],
                settings["same_fitness_percent_range"],
                settings["lenient_crossover"],
                settings["process_output_oscillators"],
                settings["verbose"],
                settings["note"]
                )
    return usersettings
end


function read_usersettings(path::String; ngenerations::Int64=-1, nspecies::Int64=3, initial_concentrations::Vector{Float64}=[1., 5., 9.], population_size::Int64=-1, seed::Int64=-1, note::String="")
    settings = Dict(
        "portion_elite" => 0.1,
        "reaction_probabilities" => [.1, .4, .4, .1],
        "p_rateconstant_mutation" => .6,                         # Probability of changning rate constant vs reaction
        "rateconstant_range" => [0.1, 50.0],
        "percent_rateconstant_change" => 20,
        "p_new_rateconstant" => 0.15,
        "population_size" => 100,
        "ngenerations" => 800,
        "nspecies" => 3,
        "nreactions" => 5,
        "max_offspring_portion" => 0.1,
        "writeout_threshold" => 0.05,
        "p_crossover" => 0,
        "p_mutation" => 1,
        "exclusive_crossover_mutation" => false,
        "portion_delete" => 0.1,
        "seed" => -1,
        "starting_delta" => 0.65,
        "delta_step" => 0.1,
        "rateconstant_distance_weight" => 0.0,
        "target_num_species" => 10,
        "use_seed_network" => false,
        "seed_network_path" => "",
        "randomize_seed_network_rates" => true,
        "tournament_select" => false,
        "chemical_species_names" => ["S0", "S1", "S2"],
        "initial_concentrations" => [1.0, 5.0, 9.0],
        "objective_data_path" => "DEFAULT",
        "match_objectivefunction_species" => true,
        "enable_speciation" => true,
        "track_fitness" => true,
        "average_fitness" => false,
        "same_fitness_crossover" => false,
        "same_fitness_percent_range" => 5,
        "lenient_crossover" => false,
        "process_output_oscillators" => false,
        "verbose" => true,
        "note"=>""
        )

    # Get the package directory
    package_dir = ""
    try
        # If using it as a package, ie location is in the julia folder
        package_dir = dirname(dirname(pathof(ReactionNetworkEvolution)))
    catch
        # Otherwise it's probably being run from a separate ReactionNetworkEvolution.jl directory
        package_dir = pwd()
    end

    use_settings_json = false
    # If theres a settings.json file there AND no other settings file is supplied, read it
    if isfile(joinpath(package_dir, "settings.json")) && path == :"DEFAULT"
        
        j = parsefile(joinpath(package_dir, "settings.json"))
        for k in keys(j)
            if k in keys(settings)
                if settings[k] != j[k]
                    settings[k] = j[k]
                    println("setting $k is different")
                    use_settings_json = true # If a non-default setting was read from this file, note it
                end
            elseif k ∉ keys(settings)
                error("$k not found in settings")
            end
        end
        if use_settings_json
            println("Reading settings from $(joinpath(package_dir, "settings.json"))")
        end
    end


    # If a path to settings is supplied:
    if path != :"DEFAULT"
        println("Reading settings from $path")
        j = parsefile(path)
        # Check for any optional args, use defaults if none 
        # If there are user specified args, replace value 
        for k in keys(j)
            if k in keys(settings)
                settings[k] = j[k]
            elseif k ∉ keys(settings)
                error("$k not found in settings")
            end
        end
    elseif !use_settings_json
        println("Using default settings")
    end
    # If seed is given as an optional arg, use it and save it to settings. 
    # This will take precedence over any seed specified in the settings file
    # If no random seed is given, check if one is specified in settings, if not, pick randomly and save it
    if seed != -1
        settings["seed"] = seed
    else
        if settings["seed"] == -1
            seed = rand(0:1000000)
            settings["seed"] = seed
        else
            seed = settings["seed"]
        end
    end
    # Set the random seed
    Random.seed!(Int64(seed))

    # Check if nspecies is supplied and update it if not default value
    # Check if the number of species names match the number of species, if not, generate them
    if nspecies != 3
        settings["nspecies"] = 3
        if length(settings["chemical_species_names"]) != nspecies
            settings["chemical_species_names"] = generate_chemical_species_names(nspecies)
        end
        settings["initial_concentrations"] = initial_concentrations
        if length(settings["initial_concentrations"]) != nspecies
            error("number of initial conditions does not match number of chemical species")
        end
    end

    # Check if the user has supplied a note
    if note != ""
        settings["note"] = note
    end


    # Get the objective function and species IDs 
    objectivefunction, floatingspeciesIDs, initial_concentrations = get_objectivefunction(settings["objective_data_path"])
    if floatingspeciesIDs != [""]
        # If using user defined objective data, get the chemical species names from there
        settings["chemical_species_names"] = floatingspeciesIDs
    end
    # If the initial conditions were read from the objective data, them put them in the settings, dict.
    # If the intital conditions are returned as [0.], then use either default or user-supplied
    if initial_concentrations != [0.] 
        settings["initial_concentrations"] = initial_concentrations
    end
    if length(initial_concentrations) != length(floatingspeciesIDs)
        error("number of initial conditions does not match number of floating species")
    end


    # Create settings object
    reaction_probabilities = ReactionProbabilities(settings["reaction_probabilities"])
    if ngenerations == -1
        ngenerations = settings["ngenerations"]
    end
    if population_size == -1
        population_size = settings["population_size"]
    end

    usersettings = Settings(settings["portion_elite"], 
                   reaction_probabilities,
                   settings["p_rateconstant_mutation"],
                   settings["rateconstant_range"],
                   settings["percent_rateconstant_change"],
                   settings["p_new_rateconstant"],
                   population_size,
                   ngenerations,
                   settings["nspecies"],
                   settings["nreactions"],
                   settings["max_offspring_portion"],
                   settings["writeout_threshold"],
                   settings["p_crossover"],
                   settings["p_mutation"],
                   settings["exclusive_crossover_mutation"],
                   settings["portion_delete"],
                   settings["seed"],
                   settings["starting_delta"],
                   settings["delta_step"],
                   settings["rateconstant_distance_weight"],
                   settings["target_num_species"],
                   settings["use_seed_network"],
                   settings["seed_network_path"],
                   settings["randomize_seed_network_rates"],
                   settings["tournament_select"],
                   settings["chemical_species_names"],
                   settings["initial_concentrations"],
                   settings["objective_data_path"],
                   settings["match_objectivefunction_species"],
                   settings["enable_speciation"],
                   settings["track_fitness"],
                   settings["average_fitness"],
                   settings["same_fitness_crossover"],
                   settings["same_fitness_percent_range"],
                   settings["lenient_crossover"],
                   settings["process_output_oscillators"],
                   settings["verbose"],
                   settings["note"],
                   )
    return usersettings, objectivefunction   
end

function writeout_settings(settings::Settings, filename::String)
    settingsdict = Dict(key=>getfield(settings, key) for key in fieldnames(Settings))
    stringsettings = json(settingsdict)
    open(filename, "w") do f
        write(f, stringsettings)
    end
end

function generate_chemical_species_names(nspecies::Int64)
    chemical_species_names = Vector{String}()
    for i in 1:nspecies
        push!(chemical_species_names, "S$i")
    end
    return chemical_species_names
end