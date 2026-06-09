using Statistics: mean #This is for calculating the mean distance for debugging
using Combinatorics: combinations # This is for looking at distances 

include("ode_solver.jl")
include("network_cleanup.jl")

mutable struct Species
    networks::Vector{ReactionNetwork}
    ID::String
    numoffspring::Int64
    speciesfitness::Float64
    topfitness::Float64
    topnetwork::ReactionNetwork
    numstagnations::Int64

    function Species(network::ReactionNetwork)
        # Start a new species with a single network
        networks = [network]
        ID = randstring(12)
        numoffspring = 0
        speciesfitness = 0
        topnetwork = network
        topfitness = 0
        numstagnations = 0
        return new(networks, ID, numoffspring, speciesfitness, topfitness, topnetwork, numstagnations)
    end

    function Species(network::ReactionNetwork, ID::String)
        # Start a new species with a single network and known ID
        networks = [network]
        numoffspring = 0
        speciesfitness = 0
        topnetwork = network
        topfitness = 0
        numstagnations = 0
        return new(networks, ID, numoffspring, speciesfitness, topfitness, topnetwork, numstagnations)
    end

    function Species(network::ReactionNetwork, ID::String, numstagnations::Int64)
        networks = [network]
        numoffspring = 0
        speciesfitness = 0
        topnetwork = network
        topfitness = 0
        return new(networks, ID, numoffspring, speciesfitness, topfitness, topnetwork, numstagnations)
    end


    function Species(listofnetworks::Vector{ReactionNetwork})
        # Start a new species with a list of networks
        networks = listofnetworks
        ID = randstring(12)
        numoffspring = 0
        speciesfitness = 0
        topnetwork = listofnetworks[1] # populate with first network, will be changed later
        topfitness = 0
        numstagnations = 0
        return new(networks, ID, numoffspring, speciesfitness, topfitness, topnetwork, numstagnations)
    end
end


function initialize_species_by_IDs(population::Array{ReactionNetwork})
    """
    This function is for creating the first species list. It takes a population of networks and assigns them 
    all to the same species. It returns a Species object."""
    species = Species(population)
    ID = species.ID
    return Dict(ID => species)
end

function get_intraspecies_distances(species_by_IDs::Dict{String, Species}, settings::Settings)
    # This measures the avg distance between individuals in a single species. 
    # It's a metric to determine how similar individuals within a species are on average
    avg_distances = []
    for ID in keys(species_by_IDs)
        species = species_by_IDs[ID]
        if length(species.networks) > 1
            distances = []
            species = species_by_IDs[ID]
            combo_indices = collect(combinations(1:length(species.networks), 2))
            for combo in combo_indices
                network1 = species.networks[combo[1]]
                network2 = species.networks[combo[2]]
                d = calculate_distance(network1, network2, settings.rateconstant_distance_weight)
                push!(distances, d)
            end
            push!(avg_distances, mean(distances))
        end
    end
    if length(avg_distances) > 0
        # This would happen if every species only had 1 individual
        return mean(avg_distances)
    else
        return -1
    end
end

function get_diversity_stats(species_by_IDs::Dict{String, Species}, settings::Settings)
    # I want to look at average distance between species and maybe also min and max.
    # I'm going to compare every species to every other species 
    all_species = collect(keys(species_by_IDs))
    combo_indices = collect(combinations(1:length(all_species),2))
    distances = []
    for combo in combo_indices
        network1 = species_by_IDs[all_species[combo[1]]].topnetwork
        network2 = species_by_IDs[all_species[combo[2]]].topnetwork
        d = calculate_distance(network1, network2, settings.rateconstant_distance_weight)
        push!(distances, d)
    end
    try
        return mean(distances), minimum(distances), maximum(distances)
    catch
        return -1, -1, -1
    end
end

function get_diversity_stats_from_list(networklist::Array{ReactionNetwork}, settings::Settings)
    combo_indices = collect(combinations(1:length(networklist),2))
    distances = []
    duplicates = []
    for combo in combo_indices
        network1 = networklist[combo[1]]
        network2 = networklist[combo[2]]
        d = calculate_distance(network1, network2, settings.rateconstant_distance_weight)
        push!(distances, d)
        if d == 0
            push!(duplicates, combo[2])
        end
    end
    duplicates = union(duplicates)
    return mean(distances), minimum(distances), maximum(distances), duplicates
end


function speciate(species_by_IDs::Dict{String, Species},
    population::Vector{ReactionNetwork},
    DELTA::Float64, 
    settings::Settings
    )
    """
    For each network, compare it to the species in the previous generation by 
    randomly picking a network in that species. If it is close enough, then it is a
    member of that species. Otherwise keep looking. If we have checked every species and it is not 
    close enough to any of them, then create a new species for it.

    If there are 2+ network that are similar to each other but dissimilar to any existing species, they will
    each be assigned to a new species. 
    """
    num_species = length(keys(species_by_IDs))
    if num_species > settings.target_num_species
        DELTA += settings.delta_step
    elseif num_species < settings.target_num_species
        DELTA -= settings.delta_step
    end

    new_species_by_IDs = Dict{String, Species}()
    for network in population
        species_assigned = false
        for speciesID in keys(species_by_IDs)
            network2 = species_by_IDs[speciesID].topnetwork# rand(species_by_IDs[speciesID].networks)
            distance = calculate_distance(network, network2, settings.rateconstant_distance_weight)
            if distance <= DELTA
                network.ID = speciesID
                # If we've already assigned a network to this species, just add it to the list 
                # Otherwise create a new entry in the hashmap and stop comparing it to existing species (break)
                if speciesID in keys(new_species_by_IDs)
                    push!(new_species_by_IDs[speciesID].networks, network)
                else
                    species = deepcopy(species_by_IDs[speciesID]) # Deepcopy the existing species to retain its data
                    species.networks = [network] # Replace the copies species networks with the new network
                    new_species_by_IDs[speciesID]  = species
                end
                # Once we've assigned a species, stop looking through the old species for a match
                species_assigned = true
                break
            end
        end
        """
        At this point there are two possible states:
            1. We have looked through the previous species, found a match, assigned the new network to it, 
            and broken the loop to land here. In this case, we just want to move onto the next network
            2. We have looked through ALL the previous species and have not found a match. The network has not bestnetwork
            assigned to a species from the previous generation and we must create a new species for it.
        """
        if !species_assigned
            newspecies = Species(network)
            new_species_by_IDs[newspecies.ID] = newspecies
        end
    end
    return new_species_by_IDs, DELTA
end

function getrandomkey(d::Dict{Vector{Vector{String}}, Reaction})
    i = rand(1:length(keys(d)))
    return collect(keys(d))[i]
end

const ACTIVE_REACTION_CAP = Ref{Union{Nothing, Int}}(nothing)

function set_active_reaction_cap!(cap::Union{Nothing, Int})
    ACTIVE_REACTION_CAP[] = cap
    return nothing
end

function active_reaction_count(network::ReactionNetwork)
    return count(r -> r.isactive, values(network.reactionlist))
end

function repair_active_reaction_cap!(network::ReactionNetwork)
    cap = ACTIVE_REACTION_CAP[]
    cap === nothing && return network
    while active_reaction_count(network) > cap
        active_keys = [key for key in keys(network.reactionlist) if network.reactionlist[key].isactive]
        isempty(active_keys) && break
        network.reactionlist[rand(active_keys)].isactive = false
    end
    return network
end


function addreaction(ng::NetworkGenerator, network::ReactionNetwork)
    cap = ACTIVE_REACTION_CAP[]
    reaction = generate_random_reaction(ng)
    if cap !== nothing && active_reaction_count(network) >= cap
        # At the cap, only accept an add if it reuses an already-active reaction.
        # Otherwise this structural mutation is rejected and the caller can retry
        # or fall back to a non-size-changing mutation.
        for _ in 1:50
            if reaction.key in keys(network.reactionlist) && network.reactionlist[reaction.key].isactive
                break
            end
            reaction = generate_random_reaction(ng)
        end
        if !(reaction.key in keys(network.reactionlist) && network.reactionlist[reaction.key].isactive)
            return network
        end
    end
    if reaction.key in keys(network.reactionlist)
        network.reactionlist[reaction.key].rateconstant += reaction.rateconstant 
    else
        network.reactionlist[reaction.key] = reaction
    end
    return network
end

function deletereaction(network::ReactionNetwork)
    if length(network.reactionlist) <= 3 # Maintain a minimum of 3 reactions
        return network
    end
    key = getrandomkey(network.reactionlist)
    # If the reaction is already inactive, this wont change anything
    network.reactionlist[key].isactive = false
    return network
end

function adddeletereaction(ng:: NetworkGenerator, network::ReactionNetwork)
    p = rand()
    if p < 0.5
        network = addreaction(ng, network)
    else
        network = deletereaction(network)
    end
    return network
end


function mutaterateconstant(network::ReactionNetwork, settings::Settings)
    key = getrandomkey(network.reactionlist) # Decide which reaction to change
    p = rand()
    if p < settings.p_new_rateconstant # Randomly pick new rate constant
        newrateconstant = rand(Uniform(settings.rateconstant_range[1], settings.rateconstant_range[2]))
        network.reactionlist[key].rateconstant = newrateconstant
        percentchange = rand(Uniform(1-settings.percent_rateconstant_change/100, 1+settings.percent_rateconstant_change/100))
        network.reactionlist[key].rateconstant *= percentchange
    else
        percentchange = rand(Uniform(1-settings.percent_rateconstant_change/100, 1+settings.percent_rateconstant_change/100))
        network.reactionlist[key].rateconstant *= percentchange
    end
    return network
end

function mutatenetwork!(settings::Settings, ng::NetworkGenerator, network::ReactionNetwork)
    p = rand()
    if p > settings.p_rateconstant_mutation
        cap = ACTIVE_REACTION_CAP[]
        if cap === nothing
            network = adddeletereaction(ng, network)
        else
            original = deepcopy(network)
            for _ in 1:20
                candidate = deepcopy(original)
                candidate = adddeletereaction(ng, candidate)
                repair_active_reaction_cap!(candidate)
                if active_reaction_count(candidate) <= cap
                    network.reactionlist = candidate.reactionlist
                    return network
                end
            end
            mutaterateconstant(network, settings)
        end
    else
        mutaterateconstant(network, settings)
    end
    repair_active_reaction_cap!(network)
    return network
end

function randomize_reaction_rates(network::ReactionNetwork, rateconstant_range::Vector{Float64})
    # Randomize the reaction rates for a given reaction network. Pick from uniform distribution 
    # within the ranges provided.
    for reactionkey in keys(network.reactionlist)
        reaction = network.reactionlist[reactionkey]
        reaction.rateconstant = rand(Uniform(rateconstant_range[1], rateconstant_range[2]))
    end
    return network
end

function generate_network_population(settings::Settings, ng::NetworkGenerator)
    population = Vector{ReactionNetwork}(undef, settings.population_size)
    if settings.use_seed_network
        seednetwork_str = load_antimony_file(settings.seed_network_path)
        seednetwork = convert_from_antimony(seednetwork_str)
        for i in 1:settings.population_size
            network = deepcopy(seednetwork)
            if settings.randomize_seed_network_rates
                network = randomize_reaction_rates(network, settings.rateconstant_range)
            end
            population[i] = network
        end
    else
        for i in 1:settings.population_size
           population[i] = generate_random_network(ng)
        end
    end

    return population
end

function sortbyfitness!(population::Array{ReactionNetwork}, rev::Bool=true)
    # If rev = true, sorts by descending
    # Use rev = true if large fitness = better
    sort!(population, by=get_fitness, rev=rev)
    return population
end

# This is for debugging mostly
function print_top_fitness(n::Int, population::Array{ReactionNetwork})
    for i in 1:n
        println(population[i].fitness)
    end
    return nothing
end

function evaluate_population_fitness(objfunct::ObjectiveFunction, species_by_IDs::Dict{String, Species}, settings::Settings)
    # Evaluates the entire populations fitness as well as the fintess for each species, per the NEAT algo
    # Assigns each individual its fitness score
    # total_fitness = sum of all fitness scores across the entire population, float
    # fitness_by_species = tracks fitness sum of all individuals within a single species, 
    #   hashmap(key, val)= speciesID, float
    # NOTE: The reason this also returns networks_by_species this hashmap is modified, the fitness score is 
    # added to each network
    total_fitness = 0.0

    for speciesID in keys(species_by_IDs)
        species = species_by_IDs[speciesID]
        species_fitness = 0
        N = length(species.networks)
        topfitness = 0
        topnetwork = species.networks[1]
        for network in species.networks
            fitness = (evaluate_fitness(objfunct, network, settings))
            # total_fitness += fitness
            species_fitness += fitness
            network.fitness = fitness
            # Check if it is the best fitness
            if fitness >= topfitness
                topfitness = fitness
                topnetwork = network
            end
        end
        if settings.average_fitness
            total_fitness += species_fitness/N
            species.speciesfitness = species_fitness/N #species fitness is now going to be the average fitness of member species
        else
            total_fitness += topfitness
            species.speciesfitness = topfitness
        end

        oldfitness = species.topfitness
        species.topfitness = topfitness
        species.topnetwork = topnetwork
        
        # If the previous top fitness is the same as this one, increment the stagnation counter by one. Otherwise, reset it to 0
        if oldfitness == topfitness
            species.numstagnations += 1
        else
            species.numstagnations = 0
        end
    end
    return species_by_IDs, total_fitness
end

function count_best_networks(bestnetwork::ReactionNetwork, population::Vector{ReactionNetwork})
    bestnetwork_count = 0
    for network in population
        if network == bestnetwork
            bestnetwork_count += 1
        end
    end
    return bestnetwork_count
end
 

function calculate_num_offspring(species_by_IDs::Dict{String, Species}, total_fitness::Float64, settings::Settings; writeoutdir::String="stalled_models/")
    # Calculates how many offspring each species gets based on its share of the total fitness (sum of all individuals)
    # Returns a hashmap key, val = speciesID, float, number of offspring for that species
    # If speciation is turned off, it will assign give the single species all of the offspring
    
    if !settings.enable_speciation
        for speciesID in keys(species_by_IDs) # There should only be 1 speciesID if enable_speciation = false
            species = species_by_IDs[speciesID]
            species.numoffspring = settings.population_size
        end
        return species_by_IDs, settings.population_size
    end

    total = settings.population_size
    total_offspring = 0 #This is how many offspring we calculate here, for debugging

    # NEW: cap number of offspring to 10% of population
    MAX_OFFSPRING = round(settings.max_offspring_portion*total)

    for speciesID in keys(species_by_IDs)
        species = species_by_IDs[speciesID]
        ##If the champion of the species has stagnated for more than 15 generations, it won't be allowed to reproduce
        if species.numstagnations >= 5000000 #disabling this for now
            species.numoffspring = 0
            networks = sortbyfitness!(species.networks) #TODO: This might not be necessary
            if networks[1].fitness > settings.writeout_threshold
                writeoutnetwork(networks[1], "HF_$(networks[1].ID).txt", directory=writeoutdir)
            else
                writeoutnetwork(networks[1], "$(networks[1].ID).txt", directory=writeoutdir)
            end
        else
            portion_offspring = species.speciesfitness/total_fitness  # The portion of the next generation this species gets to produce
            numoffspring = round(portion_offspring * total) # The number of offspring this species gets to produce
            # Cap number of offspring
            if numoffspring > MAX_OFFSPRING
                numoffspring = MAX_OFFSPRING
            end
            total_offspring += numoffspring
            species.numoffspring = numoffspring # The number of offspring this species gets to produce
        end
    end
    return species_by_IDs, Int64(total_offspring)
end

function cleanupreactions(network::ReactionNetwork)
    network = removenullreactions(network)
    reactionset = ReactionSet()
    for reaction in network.reactionlist
        reactionset = pushreaction(reactionset, reaction)
    end
    network.reactionlist = reactionset.reactionlist
    return network
end

function reproduce_networks(species_by_IDs, settings::Settings,
    ng::NetworkGenerator,
    objfunct::ObjectiveFunction,
    total_offspring::Int64)
    
    newpopulation = Vector{ReactionNetwork}(undef, total_offspring)

    offspring_index = 1

    for speciesID in keys(species_by_IDs)
        species = species_by_IDs[speciesID]
        networks = sortbyfitness!(species.networks)
        
        totaloffspring = species.numoffspring
        # If species is only allowed one offspring, take the most fit individual, mutate it, and either pass on the
        # mutated network or the original, whichever is more fit
        if totaloffspring == 1
            network = networks[1]
            newnetwork = deepcopy(networks[1])
            newnetwork = mutatenetwork!(settings, ng, newnetwork)
            newfitness = evaluate_fitness(objfunct, newnetwork, settings)
            if newfitness > network.fitness
                newpopulation[offspring_index] = newnetwork
            else
                newpopulation[offspring_index] = network
            end
            offspring_index += 1
        elseif totaloffspring > 1 # Total offspring greater than 1
            # Calculate number elite
            num_elite = Int64(round(totaloffspring*settings.portion_elite))
            if num_elite == 0 && settings.portion_elite != 0.0 # Set minimum number of elites to 1, unless the user has specifically set this value to 0
                num_elite = 1
            end
            # By basing the number of elites to copy over on the size of the subsequent generation, in some cases the number of elites to copy over
            # will be greater than the number of individuals in the current species. In this case, we copy over all the individuals in the current species,
            # and then populate the rest of the next generation with mutated networks. 
            # Possible problem: maybe this will stop evolution if it happens several times and the networks never get modified?
            if num_elite > length(networks)
                num_elite = length(networks)
            end

            for i = 1:num_elite
                newpopulation[offspring_index] = deepcopy(networks[i])
                offspring_index += 1
            end
            # Get rid of the worst networks in the species
            num_to_remove = Int64(floor(length(networks)*settings.portion_delete))
            networks = networks[1:end - num_to_remove] 
            # For the rest of the new population:
            offspring_to_add = totaloffspring - num_elite
            for i in 1:offspring_to_add
                if settings.tournament_select
                    network = tournament_select(species)
                    network2 = tournament_select(species)
                else
                    idx1, idx2 = rand(1:length(networks), 2) #TODO: this does NOT prevent same network from being selected twice, which might be good if we only have one more slot to fill
                    network = networks[idx1]
                    network2 = networks[idx2]
                end
                # Decide to mutate it, cross it over, or both
                p = rand()                
                if settings.excluisve_crossover_mutation
                    # if exclusive_crossover_mutation is set to true, then either crossover OR mutation will occur. 
                    # If p_crossover + p_mutation is < 1, then there's a chance that neither will occur.
                    if p < settings.p_crossover && idx1 != idx2
                        newnetwork = crossover(network, network2, settings)
                    elseif p < settings.p_crossover + settings.p_mutation
                        newnetwork = deepcopy(network)
                        newnetwork = mutatenetwork!(settings, ng, newnetwork)
                    else
                        newnetwork = deepcopy(network)
                    end
                else
                    # If exclusive_crossover_mutation is set to false (default), then mutation, crossover, or BOTH, will occcur
                    if p < settings.p_crossover && idx1 != idx2
                        newnetwork = crossover(network, network2, settings)
                    else
                        newnetwork = deepcopy(network)
                    end
                    if p >= 1 - settings.p_mutation
                        mutatenetwork!(settings, ng, newnetwork)
                    end
                end
                newpopulation[offspring_index] = newnetwork
                offspring_index += 1

            end
        end
    end
    return newpopulation 
end


function calculate_distance(network1::ReactionNetwork, network2::ReactionNetwork, rateconstant_distance_weight::Float64)
    # If every single reaction is different, then the distance will be 1
    num_diff = 0
    num_same = 0
    sum_differences = 0
    # rateconstant_distance_weight = 1
    # c1 = 1 # Value from paper
    # c2 = 0.4 # Value from paper (as C3 in the paper) 
    if length(network1.reactionlist) >= length(network2.reactionlist)
        largernetwork = network1
        smallernetwork = network2
    else
        largernetwork = network2
        smallernetwork = network1
    end

    N = length(largernetwork.reactionlist)

    for key in keys(largernetwork.reactionlist)
        if key ∉ keys(smallernetwork.reactionlist)
            num_diff += 1
        else # If the reaction is in both networks, look at how different the rate constants are
            sum_differences += abs(largernetwork.reactionlist[key].rateconstant - smallernetwork.reactionlist[key].rateconstant)
            num_same += 1
        end
    end

    distance = num_diff/N + (rateconstant_distance_weight*(sum_differences/num_same))
    return distance
end

function tournament_select(species::Species)
    # Select a network from a species via tournament, returns the (unmodifed) selected network
    networks = species.networks
    idx1, idx2 = rand(1:length(networks), 2)
    network1 = networks[idx1]
    network2 = networks[idx2]
    if network1.fitness > network2.fitness
        return network1
    else
        return network2
    end
end

function lenient_crossover(network1::ReactionNetwork, network2::ReactionNetwork)
    """
    A more lenient crossover method
    1. All reactions that are common are inherited with the rate constants from either parents
    2. All reactions that are only in the fit parent are inherited
    3. Reactions that are only in the less fit parent have 50% of inheritance
        
    """
    newreactiondict = Dict()
    if network1.fitness > network2.fitness
        morefitnetwork = network1
        lessfitnetwork = network2
    elseif network1.fitness < network2.fitness
        morefitnetwork = network2
        lessfitnetwork = network1
    else #If equal fitness, randomly assign roles  
        p = rand()
        if p < 0.5
            morefitnetwork = network1
            lessfitnetwork = network2
        else
            morefitnetwork = network2
            lessfitnetwork = network1
        end
    end
    # We are going to look through the genes in the more fit network. If there is a gene in the 
    # less fit network that is not in the more fit one, we don't care. But if there is an unmatched gene in the more
    # fit network, we want to keep it
    for key in keys(morefitnetwork.reactionlist)
        # If the reaction is in both networks, randomly copy it from either network
        if key in keys(lessfitnetwork.reactionlist)
            p = rand()
            if p < 0.5
                newreaction = morefitnetwork.reactionlist[key]
            else
                newreaction = lessfitnetwork.reactionlist[key]
            end

        else # If the reaction is NOT in the less fit network, copy it over
            newreaction = morefitnetwork.reactionlist[key]
        end
        # If the selected reaction is inactive, 25% of it being reactivated
        if !newreaction.isactive
            p = rand()
            if p < 0.25
                newreaction.isactive = true
            end
        end
        newreactiondict[key] = newreaction
    end
    for key in keys(lessfitnetwork.reactionlist)
        if key ∉ keys(morefitnetwork.reactionlist)
            p = rand()
            if p < 0.5
                newreaction = lessfitnetwork.reactionlist[key]
                if !newreaction.isactive
                    p_active = rand()
                    if p_active < 0.25
                        newreaction.isactive = true
                    end
                end
                newreactiondict[key] = newreaction
            end
        end
    end
    newnetwork = deepcopy(morefitnetwork)
    newnetwork.reactionlist = newreactiondict

    return newnetwork

end

function same_fitness_crossover(network1::ReactionNetwork, network2::ReactionNetwork)
    """
    A more lenient crossover method
    If networks are within 5% fitness of each other, then coin toss for all reactions

    PROBLEM: If you have two small networks, you can end not passing down ANY reactions...
    """
    newreactiondict = Dict()

    for key in keys(network1.reactionlist)
        # If the reaction is in both networks, randomly copy it from either network
        p = rand()
        if key in keys(network2.reactionlist)
            if p < 0.5
                newreactiondict[key] = network1.reactionlist[key]
            else
                newreactiondict[key] = network2.reactionlist[key]
            end
        # If the reaction is only in network1, coin toss to pass it on or not
        else
            if p < 0.5
                newreactiondict[key] = network1.reactionlist[key]
            end
        end
        # If a new reaction is added and it's inactive, .25 probability to reactivate it
        if key in keys(newreactiondict) && !newreactiondict[key].isactive && rand() < 0.25
            newreactiondict[key].isactive = true
        end
    end

    # Look for reactions that are only in network2, coin toss to pass it on or not
    for key in keys(network2.reactionlist)
        p = rand()
        if key ∉ keys(network1.reactionlist) && p < 0.5
            newreactiondict[key] = network2.reactionlist[key]
            if !newreactiondict[key].isactive && rand() < 0.25
                newreactiondict[key].isactive = true
            end
        end
    end

    newnetwork = deepcopy(network1) # It doesn't matter which one is copied because reactions will be replaced
    if length(keys(newreactiondict)) == 0 
        # If we didn't manage to copy any reactions to the offspring, just return one of the parents
        p = rand()
        if p < 0.5
            return network1
        else
            return network2
        end
    end
    newnetwork.reactionlist = newreactiondict

    return newnetwork
end

function network_fitness_is_similar(network1::ReactionNetwork, network2::ReactionNetwork, range::Float64)
    # Determines if two networks have similar fitness or not
    if network1.fitness > network2.fitness
        return network2.fitness >= (1-range/100)*network1.fitness && network2.fitness <= (1+range/100)*network1.fitness
    elseif network1.fitness < network2.fitness 
        return network1.fitness >= (1-range/100)*network2.fitness && network1.fitness <= (1+range/100)*network2.fitness
    else # networks have the same fitness
        return true
    end
end

function crossover(network1::ReactionNetwork, network2::ReactionNetwork, settings::Settings)
    if settings.same_fitness_crossover && network_fitness_is_similar(network1, network2, settings.same_fitness_percent_range)
        return same_fitness_crossover(network1, network2)
    elseif settings.lenient_crossover
        return lenient_crossover(network1, network2)
    else
        return general_crossover(network1, network2)
    end
end

function general_crossover(network1::ReactionNetwork, network2::ReactionNetwork)
    newreactiondict = Dict()
    if network1.fitness > network2.fitness
        morefitnetwork = network1
        lessfitnetwork = network2
    elseif network1.fitness < network2.fitness
        morefitnetwork = network2
        lessfitnetwork = network1
    else #If equal fitness, randomly assign roles  
        p = rand()
        if p < 0.5
            morefitnetwork = network1
            lessfitnetwork = network2
        else
            morefitnetwork = network2
            lessfitnetwork = network1
        end
    end
    # We are going to look through the genes in the more fit network. If there is a gene in the 
    # less fit network that is not in the more fit one, we don't care. But if there is an unmatched gene in the more
    # fit network, we want to keep it
    for key in keys(morefitnetwork.reactionlist)
        # If the reaction is in both networks, randomly copy it from either network
        if key in keys(lessfitnetwork.reactionlist)
            p = rand()
            if p < 0.5
                newreaction = morefitnetwork.reactionlist[key]
            else
                newreaction = lessfitnetwork.reactionlist[key]
            end
            
        else # If the reaction is NOT in the less fit network, copy it over
            newreaction = morefitnetwork.reactionlist[key]
        end
        # If the selected reaction is inactive, 25% of it being reactivated
        if !newreaction.isactive
            p = rand()
            if p < 0.25
                newreaction.isactive = true
            end
        end
        newreactiondict[key] = newreaction
    end

    newnetwork = deepcopy(morefitnetwork)
    newnetwork.reactionlist = newreactiondict

    return newnetwork

end

function writeoutnetwork(network::ReactionNetwork, filename::String, directory::String)
    astr = convert_to_antimony(network)
    astr *= "\n#fitness: $(network.fitness)"

    if !isdir(directory)
        mkdir(directory)
    end
    
    path = joinpath(directory, filename)

    open(path, "w") do file
        write(file, astr)
    close(file)
    end
end

function gettopmodel(species_by_IDs::Dict{String, Species})
    maxfitness = 0
    topnetwork = nothing
    topspecies = nothing
    for speciesID in keys(species_by_IDs)
        species = species_by_IDs[speciesID]
        if species.topfitness > maxfitness
            maxfitness = species.topfitness
            topnetwork = species.topnetwork
            topspecies = species
        end
    end
    return topnetwork, maxfitness
end
