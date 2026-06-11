using DifferentialEquations
using Sundials

include("reaction_network.jl")


function test_ode_funct(du::Array{Float64}, u::Array{Float64}, network::ReactionNetwork, t::Float64)
    # This is the same function as ode_func! but it returns du for testing purposes
    chemical_species_names = network.chemical_species_names
    
    # Reset du
    for i = 1:length(chemical_species_names)
        du[i] = 0.0
    end

    for key in keys(network.reactionlist)
        reaction = network.reactionlist[key]
        if !reaction.isactive
            continue
        end
        # Get the relevant concentrations
        dspecies = 1 # Is there a case where this would be wrong?
        for s in reaction.substrate
            idx = findfirst(item -> item == s, chemical_species_names)
            dspecies *= u[idx]
        end
        # Multiply by the rate constant to get the rate for *this* reaction
        dspecies *= reaction.rateconstant
        # Subtract this rate for substrates
        for s in reaction.substrate
            idx = findfirst(item -> item == s, chemical_species_names)
            du[idx] -= dspecies
        end
        # Add this rate for products
        for p in reaction.product
            idx = findfirst(item -> item == p, chemical_species_names)
            du[idx] += dspecies
        end
    end
    # for boundary species, reset the rate of change to 0
    for s in network.boundaryspecies
        idx = findfirst(item -> item == s, chemical_species_names)
        du[idx] = 0.0
    end
    return du
end


function ode_funct!(du::Array{Float64}, u::Array{Float64}, network::ReactionNetwork, t::Float64)
    chemical_species_names = network.chemical_species_names
    
    # Reset du
    for i = 1:length(chemical_species_names)
        du[i] = 0.0
    end

    for key in keys(network.reactionlist)
        reaction = network.reactionlist[key]
        if !reaction.isactive
            continue
        end
        # Get the relevant concentrations
        dspecies = 1 # Is there a case where this would be wrong?
        for s in reaction.substrate
            idx = findfirst(item -> item == s, chemical_species_names)
            dspecies *= u[idx]
        end
        # Multiply by the rate constant to get the rate for *this* reaction
        dspecies *= reaction.rateconstant
        # Subtract this rate for substrates
        for s in reaction.substrate
            idx = findfirst(item -> item == s, chemical_species_names)
            du[idx] -= dspecies
        end
        # Add this rate for products
        for p in reaction.product
            idx = findfirst(item -> item == p, chemical_species_names)
            du[idx] += dspecies
        end
    end
    # for boundary species, reset the rate of change to 0
    for s in network.boundaryspecies
        idx = findfirst(item -> item == s, chemical_species_names)
        du[idx] = 0.0
    end
end

function solve_ode(network::ReactionNetwork, tspan::Tuple{Float64, Float64})
    # Just solve the network with given tspan
    u0 = network.initialcondition
    ode_prob = ODEProblem(ode_funct!, u0, tspan, network)
    sol = solve(ode_prob, CVODE_BDF(), verbose=false, dense=false)#, save_everystep=false)
    return sol
end


function solve_ode(objfunct::ObjectiveFunction, network::ReactionNetwork)
    # 
    #Get time info
    t0 = first(objfunct.time)
    t_end = last(objfunct.time)
    stepsize = objfunct.time[2] - t0 # for now we assume even spacing of time points
    tspan = (t0, t_end)

    u0 = network.initialcondition
    ode_prob = ODEProblem(ode_funct!, u0, tspan, network)
    sol = solve(ode_prob, CVODE_BDF(), saveat=stepsize, verbose=false, dense=false, save_everystep=false)
    return sol
    
end


function evaluate_fitness(objfunct:: ObjectiveFunction, network::ReactionNetwork, settings::Settings)
    try 
        sol = solve_ode(objfunct, network)
        if length(sol.t) != length(objfunct.time) # If the time points are unequal, then simulation has failed.
            return 0
        end
        if settings.objective_data_path == "DEFAULT"
            fitness = zeros(length(network.floatingspecies))
            for (i, row) in enumerate(sol.u)
                for j in 1:length(network.floatingspecies)
                    fitness[j] += abs(row[j]- objfunct.objectivedata[i])
                end
            end
            return 1/(minimum(fitness))
        else
            fitness = 0
            if settings.match_objectivefunction_species
                for (i, row) in enumerate(sol.u)
                    fitness += sum(abs.(row - objfunct.objectivedata[i, :]))
                end
            end
            return 1/fitness
        end
    catch e
        return 0
    end
    
end





