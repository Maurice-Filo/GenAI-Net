using RoadRunner


function reactants_isequal(reactants1, reactants2)
    if length(reactants1) != length(reactants2)
        return false
    elseif length(reactants1) == 1
        return reactants1 == reactants2
    else
        return (reactants1[1] == reactants2[1] && reactants1[2] == reactants2[2]) ||
            (reactants1[1] == reactants2[2] && reactants1[2] == reactants2[1])

    end
end

function reaction_isnull(reaction)
    # returns true is reaction is pointless, eg. S1 -> S1
    return reactants_isequal(reaction.substrate, reaction.product)
end

function reaction_isequal(reaction1::Reaction, reaction2::Reaction)
    return reactants_isequal(reaction1.substrate, reaction2.substrate) &&
        reactants_isequal(reaction1.product, reaction2.product)    
end 

function removenullreactions(network::ReactionNetwork)
    newreactionlist = []
    for reaction in network.reactionlist
        if !reaction_isnull(reaction)
            push!(newreactionlist, reaction)
        end
    end
    network.reactionlist = newreactionlist
    return network
end

struct ReactionSet
    reactionlist:: Vector{Reaction}
    
    function ReactionSet()
        new([])
    end

end

function pushreaction(reactionset::ReactionSet, reaction; addrateconstants=true)
    for r in reactionset.reactionlist
        if reaction_isequal(r, reaction)
            if addrateconstants
                r.rateconstant += reaction.rateconstant
            end
            return reactionset
        end
    end
    push!(reactionset.reactionlist, reaction)
    return reactionset
end

# TODO: What about boundary species?
function convert_to_antimony(network::ReactionNetwork)
    reactions = ""
    rateconstants = ""
    initial_concentrations = ""
    i = 1 # This will keep track of the reaction number for the purposes of naming the rate constants, e.g. k1
    for k in keys(network.reactionlist)
        reaction = network.reactionlist[k]
        if reaction.isactive
            # Look at substrate(s) and  begin construction of reaction string and rate law string
            # For now this assumes that there are always either 1 or 2 substrates
            rxnstring = ""
            ratelawstring = ""
            if length(reaction.substrate) == 1
                rxnstring *= "$(reaction.substrate[1]) -> "
                ratelawstring *= "k$i*$(reaction.substrate[1])"
            elseif length(reaction.substrate) == 2
                rxnstring *= "$(reaction.substrate[1]) + $(reaction.substrate[2]) -> "
                ratelawstring *= "k$i*$(reaction.substrate[1])*$(reaction.substrate[2])"
            end
            # Finish the reaction and rate law strings by looking at the products
            # Again, this assumes that there are always either 1 or 2 products
            if length(reaction.product) == 1
                rxnstring *= "$(reaction.product[1]); "
            elseif length(reaction.product) == 2
                rxnstring *= "$(reaction.product[1]) + $(reaction.product[2]); "
            end
            rxnstring *= ratelawstring # Add the rate law string to the reaction string
            reactions *= "$rxnstring\n" # Add the reaction string to the string of ALL reactions
            rateconstants *= "k$i = $(reaction.rateconstant)\n" # Add rate constant to string of ALL rate constants
            i += 1
        end
    end
    
    # Now look at the species list and initial conditions to create a string of initial conditions
    for (i, s) in enumerate(network.chemical_species_names)
        initial_concentrations *= "$s = $(network.initialcondition[i])\n"
    end
    if length(network.boundaryspecies) > 0
        boundarystr = "const "
        for b in network.boundaryspecies
            boundarystr *= "$b, "
        end
        #remove the extra ", " from the String
        boundarystr = chop(boundarystr, tail=2)
        initial_concentrations *= boundarystr
    end
    astr = reactions * rateconstants * initial_concentrations # Concatenate all strings together
    return astr
end

function parse_reaction(line::String)
    # Returns reaction object and the name of the rate constant, eg. "k1"
    # Strip spaces
    line = filter(x -> !isspace(x), line)
    if occursin("->", line)
        substrates, products_ratelaw = split(line, "->")
    elseif occursin("=>", line)
        substrates, products_ratelaw = split(line, "=>")
    else
        println("invalid reaction string")
        return nothing # Not a valid reaction, hopefully this line is never called
    end
    # substrates
    s = split("+", substrates)
    products, ratelaw = split(";", products_ratelaw) #TODO: edit this if using more rate laws
    p = split("+", products)
    # Get name of rate parameter
    for subs in s
        ratelaw = replace(ratelaw, s=> "")
    end
    ratelaw = replace(ratelaw, '*'=> "")
    return Reaction(s, p, 0.0), ratelaw
end


function separate_antimony_elements(antstr::String)
    antlines = split(antstr, '\n')
    popfirst!(antlines) # get rid of "// Created by ..." line
    #TODO: make this less sketch?
    all_arrays = [[],[],[],[], []]
    index = 0
    for line in antlines
        if line == ""
            continue
        elseif startswith(line, "//")
            index += 1
            continue
        else
            push!(all_arrays[index], line)
        end
    end
    return all_arrays

end

function process_species_lines(specieslines)
    specieslines = split(specieslines[1], " ")
    chemical_species_names = Vector{String}()
    boundary = Vector{String}()
    floating = Vector{String}()
    for str in specieslines[2:end] # first item will be "species"
        str = replace(str, "," => "")
        str = replace(str, ";" => "")
        if startswith(str, "\$")
            str = str[2:end]
            push!(boundary, str)
        else
            push!(floating, str)
        end
        push!(chemical_species_names, str)
    end
    return chemical_species_names, boundary, floating
end

function substring_to_string(substring::Vector{SubString{String}})
    # When splitting a string, the result is a vector of substrings, we want a vector of strings
    newstrings = Vector{String}()
    for s in substring
        push!(newstrings, "$s")
    end
    return newstrings
end

function process_reaction_lines(reactionlines)
    #TODO: handle stoichiometric coefficients in antimony model?
    reactions_by_ratesymbol = Dict()
    for line in reactionlines
        line = replace(line, "=>" =>"->") # Make sure they all have same arrow
        line = replace(line," "=>"")
        line = replace(line, "\$" => "")
        if occursin(":", line)
            line = split(line, ":")[2]
        end # Remove reaction name
        line, ratesymbol = split(line, ";") # Remove rate law (for now)
        # Substrates and products
        substrates, products = split(line, "->")
        substrates = substring_to_string(split(substrates, "+"))
        products = substring_to_string(split(products, "+"))
        # Get the symbol for the rate constant
        ratesymbol = replace(ratesymbol,"*" => "")
        for s in substrates
            ratesymbol = replace(ratesymbol, s=>"")
        end
        reactions_by_ratesymbol[ratesymbol] = Reaction(substrates, products, 0.0) # We'll deal with rate constant later...?
    end
    return reactions_by_ratesymbol
end

function process_initialcondition_lines(iclines)
    initial_conditions = Vector{Float64}()
    for line in iclines
        line = replace(line, ";"=>"")
        line = replace(line, " "=>"")
        _, value = split(line, "=")
        push!(initial_conditions, parse(Float64, value))
    end
    return initial_conditions
end

function process_rateconstant_lines(rateconstantlines)
    rate_by_ratesymbol = Dict()
    for line in rateconstantlines
        line = replace(line, ";"=>"")
        line = replace(line, " "=>"")
        symbol, value = split(line, "=")
        rate_by_ratesymbol[symbol] = parse(Float64, value)
    end
    return rate_by_ratesymbol
end

function assign_rates_to_reaction(reactions_by_ratesymbol, rates_by_ratesymbol)
    reactionlist = Dict{Vector{Vector{String}}, Reaction}()
    for k in keys(reactions_by_ratesymbol)
        reactions_by_ratesymbol[k].rateconstant = rates_by_ratesymbol[k]
        reactionlist[reactions_by_ratesymbol[k].key] = reactions_by_ratesymbol[k]
    end
    return reactionlist
end

function get_initialcondition_values(chemical_species_names, initial_concentrations, sublist)
    sublist_initial_concentrations = Vector{Float64}()
    for s in sublist
        idx = findfirst(item -> item == s, chemical_species_names)
        push!(sublist_initial_concentrations, initial_concentrations[idx])
    end
    return sublist_initial_concentrations
end


function load_antimony_file(filepath::String)
    rawanstr = read(filepath, String)
    return rawanstr
end

function convert_from_antimony(rawantstr::String)
    # Converts an antimony string into a ReactionNetwork structure
    r = RoadRunner.loada(rawantstr)
    antstring = RoadRunner.getCurrentAntimony(r)
    # Break up lines by category. Eg. species, reactions, etc
    component_array = separate_antimony_elements(antstring)
    # Process each category
    # Get species lists and initial concenetration    
    chemical_species_names, boundaryspecies, floatingspecies = process_species_lines(component_array[1])
    all_initial_concentrations = process_initialcondition_lines(component_array[3])
    floating_initialcondtions = get_initialcondition_values(chemical_species_names, all_initial_concentrations, floatingspecies)
    boundary_initial_concentrations = get_initialcondition_values(chemical_species_names, all_initial_concentrations, boundaryspecies)
    # Get reactions and rate constants
    reactions_by_ratesymbol = process_reaction_lines(component_array[2])
    rates_by_ratesymbol = process_rateconstant_lines(component_array[4])
    reactionlist = assign_rates_to_reaction(reactions_by_ratesymbol, rates_by_ratesymbol)
    # Construct the ReactionNetwork
    network = ReactionNetwork(chemical_species_names, all_initial_concentrations, reactionlist, floatingspecies,
        boundaryspecies, floating_initialcondtions, boundary_initial_concentrations, 0.0, "seedmodel", "seedmodel")
    return network
end
    


