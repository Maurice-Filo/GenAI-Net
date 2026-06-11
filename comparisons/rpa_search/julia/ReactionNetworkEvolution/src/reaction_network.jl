using Distributions: Uniform, sample

include("settings.jl")

mutable struct Reaction
    substrate::Vector{String}
    product::Vector{String}
    rateconstant::Float64
    innovationnumber:: Int64
    isactive::Bool
    key::Vector{Vector{String}}
    
    # function Reaction(substrate, product, rateconstant, -1, true)
    #     substrate = sort(substrate)
    #     product = sort(product)
    #     return new(substrate, product, rateconstant, -1, true)
    # end

    function Reaction(substrate::Vector{String}, product::Vector{String}, rateconstant::Float64)
        substrate = sort(substrate)
        product = sort(product)
        return new(substrate, product, rateconstant, -1, true, [substrate, product])
    end

end

function Base.hash(x::Reaction, h::UInt)
    Base.hash(x.substrate, Base.hash(x.product, Base.hash(:Reaction, h)))
end
function Base.isequal(a::Reaction, b::Reaction)
    Base.isequal(a.substrate, b.substrate) && Base.isequal(a.product, b.product)
end
# Returns `false` if any two fields compare as false; otherwise, `missing` if at least
# one comparison is missing. Otherwise `true`.
# This matches the semantics of `==` for Tuple's and NamedTuple's.
function Base.:(==)(a::Reaction, b::Reaction)
    found_missing = false
    cmp = a.substrate == b.substrate
    cmp === false && return false
    if ismissing(cmp)
        found_missing = true
    end
    cmp = a.product == b.product
    cmp === false && return false
    if ismissing(cmp)
        found_missing = true
    end
    found_missing && return missing
    return true
end

# function get_reaction_key(reaction::Reaction)
#     return [substrate, product]
# end


mutable struct ReactionNetwork
    chemical_species_names::Vector{String}
    initialcondition::Vector{Float64}
    reactionlist::Dict{Vector{Vector{String}},Reaction}
    floatingspecies::Vector{String}
    boundaryspecies::Vector{String}
    float_initialcondition::Vector{Float64}
    boundary_initialcondition::Vector{Float64}
     # This is for development
    fitness::Float64
    ID::String
    species::String

    # function ReactionNetwork(chemical_species_names::Vector{String},initialcondition::Vector{Float64},reactionlist::Vector{Reaction},floatingspecies::Vector{String},boundaryspecies::Vector{String},float_initialcondition::Vector{Float64},boundary_initialcondition::Vector{Float64})
    #     return new(chemical_species_names, initialcondition, floatingspecies, boundaryspecies, float_initialcondition,boundary_initialcondition, 1.0E8)
    # end
end

function Base.hash(x::ReactionNetwork, h::UInt)
    Base.hash(x.reactionlist, Base.hash(:ReactionNetwork, h))
end
function Base.isequal(a::ReactionNetwork, b::ReactionNetwork)
    Base.isequal(a.reactionlist, b.reactionlist)
end
# Returns `false` if any two fields compare as false; otherwise, `missing` if at least
# one comparison is missing. Otherwise `true`.
# This matches the semantics of `==` for Tuple's and NamedTuple's.
function Base.:(==)(a::ReactionNetwork, b::ReactionNetwork)
    found_missing = false
    cmp = a.reactionlist == b.reactionlist
    cmp === false && return false
    if ismissing(cmp)
        found_missing = true
    end
    found_missing && return missing
    return true
end

#TODO: floating and boundary species
struct NetworkGenerator
    chemical_species_names::Vector{String}
    initialcondition::Vector{Float64}
    numreactions::Int
    reaction_probabilities::ReactionProbabilities
    rateconstant_range::Vector{Float64}
    # seedmodel::Nothing # Change this if you want to use seed models #::Union{ReactionNetwork, Nothing}

    
    function NetworkGenerator(chemical_species_names::Vector{String}, initialcondition::Vector{Float64}, numreactions::Int, reaction_probabilities::ReactionProbabilities, rateconstant_range::Vector{Float64})#; seed=nothing)
        new(chemical_species_names, initialcondition, numreactions, reaction_probabilities, rateconstant_range)
    end

    
end

function get_networkgenerator(settings::Settings; seed=Nothing)#::NetworkGenerator
    chemical_species_names = settings.chemical_species_names
    initial_concentrations = settings.initial_concentrations
    nreactions = settings.nreactions
    reaction_probabilities = settings.reaction_probabilities
    rateconstant_range = settings.rateconstant_range

    return NetworkGenerator(chemical_species_names, initial_concentrations, nreactions, reaction_probabilities, rateconstant_range)#, seed=seed)
   
end

function get_fitness(network::ReactionNetwork)
    return network.fitness
end

#TODO: prevent pointless reactions and ensure connectivity
function generate_random_reaction(ng::NetworkGenerator)
    p = rand()
    k = rand(Uniform(ng.rateconstant_range[1], ng.rateconstant_range[2]))
    
    selectedspecies = sample(ng.chemical_species_names,4)

    if p <= ng.reaction_probabilities.uniuni #uniuni reaction
        subs = [selectedspecies[1]]
        prods = [selectedspecies[2]]
    elseif p > ng.reaction_probabilities.uniuni && p <= (ng.reaction_probabilities.uniuni + ng.reaction_probabilities.unibi) # unibi
        subs = [selectedspecies[1]]
        prods = selectedspecies[2:3]
    elseif p > (ng.reaction_probabilities.uniuni + ng.reaction_probabilities.unibi) && p <= 1 - ng.reaction_probabilities.bibi #biuni
        subs = selectedspecies[1:2]
        prods = [selectedspecies[3]]
    else #bibi
        subs = selectedspecies[1:2]
        prods = selectedspecies[3:4]
    end
    reaction = Reaction(subs, prods, k)
    return reaction
end

function generate_reactionlist(ng::NetworkGenerator)
    #TODO: What is going on here with global innovation number 
    # global global_innovation_number

    reactionlist = Dict{Vector{Vector{String}},Reaction}()
    #reactionlist = Vector{Reaction}()
    for i in 1:ng.numreactions
        reaction = generate_random_reaction(ng)
        # if reaction.key in keys(current_innovation_num_by_reaction)
        #     reaction.innovationnumber = current_innovation_num_by_reaction[reaction.key]
        # else
        #     reaction.innovationnumber = global_innovation_number
        #     global_innovation_number += 1
        #     current_innovation_num_by_reaction[reaction.key] = reaction.innovationnumber
        # end
        if reaction.key in keys(reactionlist)
            reactionlist[reaction.key].rateconstant += reaction.rateconstant
        else
            reactionlist[reaction.key] = reaction
        end
        #push!(reactionlist, reaction)
    end
    return reactionlist
end

#TODO: Make this more realistic? Maybe a poisson distribution? Or maybe in real life the user will know what species are boundaries?
#TODO: maybe the number of possible boundary species depends on how many species are in the network?
# FOR NOW: I'm just not going to deal with boundary species
function choose_boundary_species(ng::NetworkGenerator)
    # Decide how many boundary species there will be, 0 to 3, but rarely 3
    # weights = [1, 0, 0]
    # nboundary = wsample(0:2, weights)
    # Decide which will be the boundary species
    boundaryspecies = Vector{String}()#sample(ng.chemical_species_names, nboundary, replace=false)
    floatingspecies = [s for s in ng.chemical_species_names if !(s in boundaryspecies)]
    return floatingspecies, boundaryspecies
end

#TODO: Might not need to do this for the floats?
function get_initial_concentrations(ng::NetworkGenerator, floatingspecies::Vector{String}, boundaryspecies::Vector{String})
    float_initialcondition = Vector{Float64}()
    boundary_initialcondition = Vector{Float64}()
    for species in floatingspecies
        idx = findfirst(item -> item == species, ng.chemical_species_names)
        push!(float_initialcondition, ng.initialcondition[idx])
    end
    for species in boundaryspecies
        idx = findfirst(item -> item == species, ng.chemical_species_names)
        push!(boundary_initialcondition, ng.initialcondition[idx])
    end
    return float_initialcondition, boundary_initialcondition
end

function generate_random_network(ng::NetworkGenerator)
    floatingspecies, boundaryspecies = choose_boundary_species(ng)
    float_initialcondition, boundary_initialcondition = get_initial_concentrations(ng, floatingspecies, boundaryspecies)
    reactionlist = generate_reactionlist(ng)
    ID = randstring(10)
    return ReactionNetwork(ng.chemical_species_names, ng.initialcondition, reactionlist, floatingspecies, boundaryspecies, float_initialcondition, boundary_initialcondition, 0, ID, "0")
end
