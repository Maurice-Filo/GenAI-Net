using Pkg
Pkg.activate(".")
using ReactionNetworkEvolution
using ArgParse



function parse_initial_concentrations(initial_concentrations::String)
    # Strip whitespace
    initial_concentrations = filter(x -> !isspace(x), initial_concentrations)
    # Strip brackets
    initial_concentrations = initial_concentrations[2:end-1]
    # Separate by commas 
    initial_concentrations = split(initial_concentrations, ",")
    new_array = Vector{Float64}()
    for num in initial_concentrations
        push!(new_array, parse(Float64, num))
    end
    return new_array
end


s = ArgParseSettings()

@add_arg_table s begin
    "--ngenerations"
        arg_type = Int
        default = -1
    "--ntrials"
        arg_type = Int
        default = -1
    "--nspecies"
        arg_type = Int
        default = 3
    "--initial_concentrations"
        arg_type = String
        default = "[1., 5., 9.]"
    "--population_size"
        arg_type = Int
        default = -1
    "--pathtosettings"
        arg_type = String
        default = ""
    "--outputpath"
        arg_type = String
        default=""
    "--seed"
        arg_type = Int
        default = -1
    "--note"
        arg_type = String
        default = ""
end

parsed_args = parse_args(ARGS, s)

parsed_args["initial_concentrations"] = parse_initial_concentrations(parsed_args["initial_concentrations"])

ReactionNetworkEvolution.run_evolution(ngenerations=parsed_args["ngenerations"],
              ntrials=parsed_args["ntrials"],
              nspecies=parsed_args["nspecies"],
              initial_concentrations = parsed_args["initial_concentrations"],
              population_size=parsed_args["population_size"],
              pathtosettings=parsed_args["pathtosettings"],
              outputpath=parsed_args["outputpath"],
              seed=parsed_args["seed"],
              note=parsed_args["note"])
