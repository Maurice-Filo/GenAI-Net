using RoadRunner

function has_oscillator_eigens(eigen_array)
    for i in 1:size(eigen_array)[1]
        real = eigen_array[i, 1]
        imag = eigen_array[i, 2]
        if real >= 0 && imag != 0
            return true
        end
    end
    return false
end

function fix_broken_oscillator(astr::String)
    # Sometimes models that have oscillator Eigen values but negative steady state concentrations
    # can be fixed by removing a single reaction
    astr_lines = split(astr, "\n") 
    for (i, line) in enumerate(astr_lines)
        # comment out reaction
        if occursin("->", line)
            astr_lines[i] = "#" * astr_lines[i]
            new_astr = join(astr_lines, "\n")
            if is_oscillator(new_astr)
                return new_astr #???
            end
            # If that didn't work, change it back
            astr_lines[i] = astr_lines[i][2:end]
        end
    end
    return "FAIL"
end

function is_broken_oscillator(astr::String)
    # Sometimes models that have oscillator Eigen values but negative steady state concentrations
    # can be fixed by removing a single reaction
    r = RoadRunner.loada(astr)
    try
        RoadRunner.steadyState(r) # WTF
        eigens = RoadRunner.getEigenvalues(r)
        concentrations = RoadRunner.getFloatingSpeciesConcentrations(r)
        # If it has the correct eigens and positive concentrations, return true
        return has_oscillator_eigens(eigens) && !all(>=(0), concentrations)
    catch 
    end
    return false
end

function is_oscillator(astr::String)
    r = RoadRunner.loada(astr)
    try
        s = RoadRunner.steadyState(r) 
        eigens = RoadRunner.getEigenvalues(r)
        concentrations = RoadRunner.getFloatingSpeciesConcentrations(r)
        # If it has the correct eigens and positive concentrations, return true
        # But if it doesn't, then it still might be an oscillator
        if has_oscillator_eigens(eigens) && all(>=(0), concentrations)
            return true
        end
    catch 
        # Sometimes it will error when calculating the eigens or steady state, but that doesn't necessarily mean
        # it is not an oscillator
    end
    # If it fails to get eigens values, try to simulate it
    # If the simulation fails, adjust tolerances
    try
        RoadRunner.resetToOrigin(r)
        RoadRunner.setTimeStart(r, 0)
        RoadRunner.setTimeEnd(r, 50)
        RoadRunner.setNumPoints(r, 100000)
        RoadRunner.simulate(r)
    catch
        try # Try to simulate with adjusted tolerance
            RoadRunner.resetToOrigin(r)
            RoadRunner.setTimeStart(r, 0)
            RoadRunner.setTimeEnd(r, 50)
            RoadRunner.setNumPoints(r, 100000)
            RoadRunner.setCurrentIntegratorParameterString(r, "relative_tolerance", 1e-10)
            RoadRunner.simulate(r)
        catch e# if that doesn't work either, it's not an oscillator
            println(e)
            return false
        end
    end
    # Try to get the eigens again, if that doesn't work, it's not an oscillator
    try
        eigens = RoadRunner.getEigenvalues(r)
        #Now check if the eigen values indicate an oscillator
        # if it does have oscillator eigens, it's an oscillator
        if has_oscillator_eigens(eigens)
            return true
        end
    catch 
        # return false
    end
    # 
    # If that still didn't work, calculate steady state again
    try
        s = RoadRunner.steadyState(r)
        change = RoadRunner.getRatesOfChange(r)
        eigens = RoadRunner.getEigenvalues(r)
        concentrations = RoadRunner.getFloatingSpeciesConcentrations(r)
        return has_oscillator_eigens(eigens) && all(>=(0), concentrations)
    catch e
        # If after all that, something still doesn't work, then it's not an oscillator
        println(e)
        return false
    end
end

function make_output_dirs(outputpath::String)
    if !isdir(joinpath(outputpath, "SUCCESS"))
        mkdir(joinpath(outputpath, "SUCCESS"))
    end
    if !isdir(joinpath(outputpath, "FAIL"))
        mkdir(joinpath(outputpath, "FAIL"))
    end
end

function process_oscillators(outputpath::String; track_fitness::Bool=false)
    make_output_dirs(outputpath)
    for model_file in readdir(outputpath)
        if !startswith(model_file, "SUCCESS") &&  !startswith(model_file, "FAIL") && !endswith(model_file, ".json") # Ignore results folders and json files
            astr = load_antimony_file(joinpath(outputpath,  model_file))
            is_oscillator_model = is_oscillator(astr)
            # If it's an oscillator, move it to the SUCCESS dir
            if is_oscillator_model
                destination = joinpath(outputpath, "SUCCESS")
            else
                # Try to fix it 
                if is_broken_oscillator(astr)
                    astr = fix_broken_oscillator(astr)
                    if astr != "FAIL" # If fixing it worked, overwrite file
                        model_file_path = joinpath(outputpath, model_file)
                        rm(model_file_path) # Remove the old filepath
                        open(model_file_path, "w") do file
                            write(file, astr)    
                        end
                        # Fixed file will be moved to the successful results directory
                        destination = joinpath(outputpath, "SUCCESS")
                    end
                end
                destination = joinpath(outputpath, "FAIL")
            end
            mv(joinpath(outputpath, model_file), joinpath(destination, model_file), force=true)
            if track_fitness # also move the associated json fitness tracking file
                fitness_file = model_file[1:end-4] * "_fitness.json" # replace .ant with .json
                mv(joinpath(outputpath, fitness_file), joinpath(destination, fitness_file), force=true)
            end
        end
    end
    println("Output written to $outputpath")
end


# astr = "species S1, S0, S2;

# // Reactions:
# _J0: S1 -> S1 + S1; k1*S1;
# _J1: S0 + S2 -> S2; k2*S0*S2;
# _J2: S0 + S1 -> S0 + S0; k3*S0*S1;

# // Species initializations:
# S1 = 5;
# S0 = 1;
# S2 = 9;

# // Variable initializations:
# k1 = 12.4542964046187;
# k2 = 15.3768991659621;
# k3 = 7.31188654436819;

# // Other declarations:
# const k1, k2, k3;

# #fitness: 0.03963733850562187"

# println(is_oscillator(astr))