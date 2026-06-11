import tellurium as te
import math
import os
import matplotlib
import json

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def gather_best_models(path, destination):
    """
    Get models from numerous subdirectories and put all the antimony files in one place.
    Useful if the models were generated using the previous default write out which put final models under several
    layers of subdirectories
    :param path: (str) Path to the parent directory containing subdirectories for each model.
        E.g. batch_2024-05-31_122344xyz or results_20240531_122344
    :param destination: (str) Path to directory where models will be moved
    :return: None
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    if os.path.basename(path).startswith("batch"):
        fullpath = path
        # if os.path.isdir(os.path.join(path, dir, dir)):
        #     fullpath = os.path.join(path, dir, dir)
        # else:
        #     fullpath = os.path.join(path, dir)
    elif os.path.basename(path).startswith("results"):
        fullpath = os.path.join(path, "SUCCESS")
    rename_and_move_models(fullpath, destination)


def rename_and_move_models(fullpath, destination):
    """
    Private function to help in gathering the best models
    :param path: (str) path with final_models directory two layers below it.
        eg. batch_2024-05-31_122344xyz for batch_2024-05-31_122344xyz/batch_2024-05-31_122344xyz/final_models
        or results_20240531_122344 for results_20240531_122344/20240522_121431rK3/final_models
    :param destination: (str) path to directory where models will be moved.
    :return: None
    """
    for dir in os.listdir(fullpath):
        try:
            model_dir = os.path.join(fullpath, dir, "final_models")
            for model in os.listdir(model_dir):
                if model.startswith("bestmodel"):
                    name = model.split("_")[1]  # Remove "bestmodel" from name
                    if not model.endswith(".ant"):
                        name += ".ant"  # add the antimony extension if not there already
                    os.rename(os.path.join(model_dir, model), os.path.join(destination, name))
                    # os.replace(os.path.join(model_dir, name), os.path.join(destination, name))
                    break
        except FileNotFoundError:
            pass


def load_model(path):
    '''
    Loads a RoadRunner model from an antimony file.
    :param path (str): Path to the antimony text file containing the mdoel
    :return: r (RoadRunner model): RoadRunner model of the antimony string
    '''
    with open(path, "r") as f:
        astr = f.read()
    r = te.loada(astr)
    return r


#--------------------------------------------------
# Network Sorting
#-------------------------------------------------

def get_model_fitness_from_file(path):
    '''
    Extracts the fitness from an antimony file
    :param path: (str) Path to the antimony text file containing the mdoel
    :return: fitness (float) Fitness of the model
    '''
    with open(path, "r") as f:
        astr = f.read()
    for line in astr.split("\n"):
        if line.startswith("#fitness"):
            fitness = line.split(" ")[1]
            return float(fitness)


def get_model_fitness_from_antimony(astr):
    '''
    Extracts the fitness from an antimony string
    :param astr: (str) An antimony string
    :return: (float) Fitness of the model
    '''
    for line in astr.split("\n"):
        if line.startswith("#fitness"):
            fitness = line.split(" ")[1]
            return float(fitness)


def get_model_fitness(input):
    '''
    Extracts the fitness of a model
    :param input: (str) Either an antimony string or the path to an antimony file.
    :return: (float) Fitness of the model
    '''
    if "->" in input:
        return get_model_fitness_from_antimony(astr)
    else:
        return get_model_fitness_from_file(input)


def fix_model(astr, fitness=None):
    '''
    Attempts to fix a model to make it an oscillator by removing one reaction at a time.
    :param astr: (str) Antimony string of the model
    :param fitness: (float) Fitness of the model (optional). If provided, will be appended to the new antimony string
    :return: (bool) if the model was successfully fixed, (str) The new antimony string if the model was successfully
    fixed, otherwise it returns the input antimony string.
    '''
    split_astr = astr.split("\n")
    for i in range(len(split_astr)):
        if "->" in split_astr[i]:
            split_astr[i] = "#" + split_astr[i]
            newastr = "\n".join(split_astr)
            r = te.loada(newastr)
            if is_oscillator(r):
                if fitness:
                    newastr.append(f"\n# fitness: {fitness}")
                return True, newastr
            # If the model is not fixed, then uncomment the line
            split_astr[i] = split_astr[i][1:]
    # If the model can't be fixed, return None
    if fitness:
        newastr.append(f"\n# fitness: {fitness}")
    return False, astr


def check_eigens(eigen_array):
    '''
    Check if the eigenvalues of the model indicate sustained oscillation
    :param eigen_array: (numpy.ndarray) Eigen values returned from r.getFullEigenValues()
    :return: (bool) True if the eigenvalues are indicative of an oscillator
    '''
    for num in eigen_array:
        if num.real >= 0 and num.imag != 0:
            return True
    return False


def is_broken_oscillator(r):
    """
    Test if the model has good eigen values but subzero concentrations
    Often these models can be fixed by removing one reaction
    Return (is_oscillator, is_broken_oscillator)
    If we do this test and find a good oscillator, then we'll return true and skip examining it again
    :param r: A roadrunner model)
    :return: (bool, bool)
    (True, False) -> Add it to list of successful oscillators
    (False, True) -> Add it to list of models to be fixed
    (False, False) -> Proceed with further checks for oscillation
    """
    is_broken = False
    is_good_oscillator = False

    try:
        s = r.steadyState()
        eigens = r.getFullEigenValues()
        hasgoodvals = check_eigens(eigens)
        if hasgoodvals:
            if not all(i >= 0 for i in r.getFloatingSpeciesConcentrations()):
                is_broken = True
            else:
                is_good_oscillator = True
    except:
        pass
    return is_good_oscillator, is_broken


def is_oscillator_preprocessed(r):
    """
    This is the "private" is_oscillator function. It assumes preprocessing has already been done with
    the function is_broken_oscillator and skips finding the inital steady state eigenvalues.
    :param r: Roadrunner model
    :return: (bool) True is the model is an oscillator.
    """
    r.resetToOrigin()
    try:
        r.simulate(0, 50, 100000)

    except:
        try:
            r.integrator.relative_tolerance = 1e-10
            r.resetToOrigin()
            r.simulate(0, 50, 100000)
        except:
            return False
    try:
        eigens = r.getFullEigenValues()
    except:
        return False
    hasgoodvals = check_eigens(eigens)
    if hasgoodvals:
        return True
    # 4. If that still didn't work, calculate steady state again
    try:
        s = r.steadyState()
        change = r.getRatesOfChange()
        eigens = r.getFullEigenValues()
        hasgoodvals = check_eigens(eigens)
    except:
        pass
    # And if that doesn't work then it's definitely (?) not an oscillator
    return hasgoodvals and all(i >= 0 for i in r.getFloatingSpeciesConcentrations())


def is_oscillator(r):
    """
    This is intended to be the "public" function. Given a roadrunner model, perform all tests for oscillation.
    It will mark "broken" oscillators as non-oscillators and will not attempt repair
    :param r: RoadRunner model
    :return: (bool) True is the model oscillates
    """
    try:
        r.resetToOrigin()
        s = r.steadyState()
        eigens = r.getFullEigenValues()
        hasgoodvals = check_eigens(eigens)
        if hasgoodvals and all(i >= 0 for i in r.getFloatingSpeciesConcentrations()):
            return True
    except:
        pass
    # If that didn't work, simulate for a bit
    r.resetToOrigin()
    try:
        r.simulate(0, 50, 100000)

    except:
        try:
            r.integrator.relative_tolerance = 1e-10
            r.resetToOrigin()
            r.simulate(0, 50, 100000)
        except:
            return False
    try:
        eigens = r.getFullEigenValues()
    except:
        return False
    hasgoodvals = check_eigens(eigens)
    if hasgoodvals:
        return True
    # If that still didn't work, calculate steady state again
    try:
        s = r.steadyState()
        change = r.getRatesOfChange()
        eigens = r.getFullEigenValues()
        hasgoodvals = check_eigens(eigens)
    except:
        pass
    # And if that doesn't work then it's definitely (?) not an oscillator
    return hasgoodvals and all(i >= 0 for i in r.getFloatingSpeciesConcentrations())


def evaluate_oscillators(path: str):
    """
    Evaluate models in a directly and label them as oscillators (success) or non-oscillators (fail). Repair will be
    attempted for broken oscillators.
    :param path: (str) Path to a directory of antimony models
    :return: (int, int) The number of oscillators found and the total number of models evaluated
    """
    success_count = 0
    total_count = 0
    for file in os.listdir(path):
        if not file.endswith(".json"):
            r = load_model(os.path.join(path, file))
            fitness = get_model_fitness(os.path.join(path, file))
            total_count += 1
            is_good_oscillator, is_broken = is_broken_oscillator(r)
            if is_good_oscillator:
                success_count += 1
                if "success" not in file:
                    os.rename(os.path.join(path, file), os.path.join(path, f"success_{file}"))
            elif is_broken:
                # Try to fix the broken oscillator
                is_fixed, astr = fix_model(r.getAntimony(), fitness=fitness)
                if is_fixed:
                    success_count += 1
                    with open(os.path.join(path, file), 'w') as f:
                        f.write(astr)
                    if "success" not in file:
                        os.rename(os.path.join(path, file), os.path.join(path, f"success_{file}"))
            else:
                if is_oscillator_preprocessed(r):
                    success_count += 1
                    if "success" not in file:
                        os.rename(os.path.join(path, file), os.path.join(path, f"success_{file}"))
                else:
                    if "fail" not in file:
                        os.rename(os.path.join(path, file), os.path.join(path, f"fail_{file}"))
    print(f"Processed {total_count} models and found {success_count} oscillators.")
    print(f"Success rate: {(success_count / total_count) * 100}%")
    return success_count, total_count


def sort_by_fitness(path, reverse=True):
    """
    Given a directory of models, sort them by fitness, best to worst
    :param path: (str) path to directory containing antimony files
    :param reverse: (bool, optional) If True, models will be sorted best to worst
    :return: None
    """
    model_list = []
    for file in os.listdir(path):
        fitness = get_model_fitness(os.path.join(path, file))
        model_list.append((fitness, file))
    model_list = sorted(model_list, reverse=reverse)
    for i, item in enumerate(model_list):
        os.rename(os.path.join(path, item[1]), os.path.join(path, f"{i}_{item[1]}"))
    print(f"Sorted models in {path} by fitness")


def evaluate_fitness_cutoff(path, cutoff):
    """
    Sort models as presumed oscillators (success) or presumed non-oscillators (fail) based on their fitness values.
    Models that have been repaired will be evaluated based on their pre-repair fitness value if present.
    Ignores models that don't have a fitness model
    :param path: Path to directory containing antimony models
    :param cutoff: (float) Fitness values equal or greater to this number will be considered oscillators
    :return: (int, int) The number of oscillators (successes) and the total number of models evaluated
    """
    success_count = 0
    total_count = 0
    no_fitness_count = 0
    for file in os.listdir(path):
        fitness = get_model_fitness(os.path.join(path, file))
        total_count += 1
        if fitness >= cutoff:
            success_count += 1
            if "success" not in file:
                os.rename(os.path.join(path, file), os.path.join(path, f"success_{file}"))
        elif fitness < cutoff:
            if "fail" not in file:
                os.rename(os.path.join(path, file), os.path.join(path, f"fail_{file}"))
        else:  # If no fitness value is present, ignore it
            no_fitness_count += 1
    print(f"Processed models in {path} by fitness cutoff of {cutoff}")
    if no_fitness_count == 1:
        print(f"{no_fitness_count} model did not have a fitness value and was ignored")
    elif no_fitness_count > 1:
        print(f"{no_fitness_count} models did not have fitness values and were ignored")
    print(f"{success_count} out of {total_count} models met fitness cutoff")
    return success_count, total_count


#--------------------------------------------------
# Plotting Timeseries Data
#-------------------------------------------------
def get_best_dimensions(n):
    """
    Given a number of models to plot, determine the optimal dimensions for subplot layout
    :param n: (int) the number of models to plot
    :return: (int, int) The number of rows and columns for the subplots
    """
    sqrt = n ** 0.5
    closest_2 = [math.floor(sqrt) ** 2, math.ceil(sqrt) ** 2]
    differences = [abs(x - n) for x in closest_2]
    closest = closest_2[differences.index(min(differences))]
    best = int(closest ** 0.5)
    if closest >= n:
        return best, best
    else:
        return best, best + 1


def plot_timeseries_path(path, start, end, numpoints, savepath):
    """
    Private function for plotting time series data given a path to a directory
    :param path: (str) Path to a directory containing antimony files
    :param start: (float) Starting time for simulation
    :param end: (float) Ending time for simulation
    :param numpoints: (int) Number of points for the simulation
    :param savepath: (str) Path to save the figure, if applicable
    :return:
    """
    plt.clf()
    plt.rcParams.update({'font.size': 6})
    filenames = os.listdir(path)
    n = len(filenames)
    rows, cols = get_best_dimensions(n)
    models = []
    for file in os.listdir(path):
        r = load_model(os.path.join(path, file))
        models.append(r)
    idx = 0
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            r = models[idx]
            m = r.simulate(start, end, numpoints)
            axs[i, j].plot(m['time'], m[:, 1:])
            axs[i, j].set_title(filenames[idx])
            idx += 1
            if idx > n - 1:
                break
        if idx > n - 1:
            break
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Plots saved to {savepath}")
        plt.close()
    else:
        plt.show()


def plot_timeseries_model_list(model_list, start, end, numpoints, savepath):
    """
    Private function for plotting time series data given a list of models
    :param path: list(str) OR list(RoadRunnermodels) models to plot
    :param start: (float) Starting time for simulation
    :param end: (float) Ending time for simulation
    :param numpoints: (int) Number of points for the simulation
    :param savepath: (str) Path to save the figure, if applicable
    :return: None
    """
    plt.clf()
    if isinstance(model_list[0], str):  # If it's a list of antimony strings:
        models = []
        for astr in model_list:
            r = te.loada(astr)
            models.append(r)
    else:
        models = model_list
    # Otherwise presumably it's a list of roadrunner models
    n = len(models)
    rows, cols = get_best_dimensions(n)
    idx = 0
    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            r = models[idx]
            m = r.simulate(start, end, numpoints)
            axs[i, j].plot(m['time'], m[:, 1:])
            axs[i, j].set_title(idx)
            idx += 1
            if idx > n - 1:
                break
        if idx > n - 1:
            break
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Plots saved to {savepath}")
        plt.close()
    else:
        plt.show()


def plot_single_model(model, start, end, numpoints, savepath):
    """
    Private function, plot a single model
    :param path: (str) antimony model string OR RoadRunner model to plot
    :param start: (float) Starting time for simulation
    :param end: (float) Ending time for simulation
    :param numpoints: (int) Number of points for the simulation
    :param savepath: (str) Path to save the figure, if applicable
    :return: None
    """
    plt.clf()
    if isinstance(model, str):  # if it's an antimony model
        r = te.loada(model)
    else:  #  If it's already a roadrunner model
        r = model
    m = r.simulate(start, end, numpoints)
    r.plot(savefig=savepath)


def plot_timeseries(input, start=0, end=1, numpoints=200, savepath=None):
    """
    Plot timeseries data for models
    :param input: Any of:
        1. (str) path to a directory containing antimony files
        2. List of antimony strings OR list of RoadRunner models
        3. A single antimony string OR a single RoadRunner model
    :param start: optional (float), starting time for simulation, default=0
    :param end:  optional (float), end time for simulation, default=1
    :param numpoints: optional (int), number of points for simulation, default=200
    :param savepath: optional (str), path to save the figure
    :return: None
    """
    if isinstance(input, str) and "->" not in input:  # If we're given a path
        plot_timeseries_path(input, start, end, numpoints, savepath)
    elif isinstance(input, list):  # List of models, either as antimony strings, or roadrunner models
        plot_timeseries_model_list(input, start, end, numpoints, savepath)
    else:  # antimony string for single model or single roadrunner model
        plot_single_model(input, start, end, numpoints, savepath)


#--------------------------------------------------
# Plotting Fitness Data
#--------------------------------------------------

def load_fitness_values(path):
    """
    Load a list of the fitness values for each generation
    :param path: Path to a json file containing the fitness values
    :return: list(float) fitness values for each generation
    """
    if not path.endswith(".json"):
        raise ValueError("file type must be .json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["top_individual_fitness"]


def load_many_fitness_values(path):
    """
    Load a collection of fitness values for each generation for several models
    :param path: (str) Path to a directory containing .json files
    :return: list(list(float)) fitness values for each model for each generation
    """
    many_fitness_values = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            many_fitness_values.append(load_fitness_values(os.path.join(path, file)))
    return many_fitness_values


def plot_fitness(path, limit=None, savepath=None):
    """
    Plot the fitness over time of one or more models
    :param path: (str) Either the path to a single json file or the path to a directory containing several json files
    :param limit: (int) The maximum number of time series to plot. None by default
    :param savepath: optional (str), path to save the figure
    :return: None
    """
    if os.path.isdir(path):
        plot_fitness_dir(path, limit=limit, savepath=savepath)
    else:
        plot_individual_fitness(path, savepath=savepath)


def plot_individual_fitness(path, savepath=None):
    """
    Private function. Plot the fitness over time for a single model
    :param path: (str) Path to the json file containing the fitness values
    :param savepath: optional (str), path to save the figure
    :return: None
    """
    plt.clf()
    fitness_values = load_fitness_values(path)
    plt.plot(fitness_values)
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.title("Top Fitness Trajectory")
    if savepath:
        plt.savefig(savepath)
        print(f"Plots saved to {savepath}")
        plt.close()
    else:
        plt.show()


def plot_fitness_dir(path, limit=None, savepath=None):
    """
    Private function. Plot the fitness over time for several models
    :param path: (str) Path to the directory containing the json files
    :param savepath: optional (str), path to save the figure
    :return: None
    """
    plt.clf()
    fitness_values_list = load_many_fitness_values(path)
    count = 0
    for values in fitness_values_list:
        plt.plot(values)
        count += 1
        if limit is not None and count >= limit:
            break
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    if len(fitness_values_list) == 1:
        plt.title("Top Fitness Trajectory")
    else:
        plt.title("Top Fitness Trajectories")
    if savepath:
        plt.savefig(savepath)
        print(f"Plots saved to {savepath}")
        plt.close()
    else:
        plt.show()


#------------------------------------------------------------------
# Model Cleanup
#------------------------------------------------------------------

def prune_antimony_model(astr):
    """
    Remove reactions that don't contribute to oscillation
    :param astr: (str) an antimony string
    :param fitness: optional (float), append the ORIGINAL fitness value to the new antimony string
    :return: (int, str), The number of reactions removed and the new antimony string
    """
    reactions_pruned = 0
    split_astr = astr.split("\n")
    for i in range(len(split_astr)):
        if "->" in split_astr[i]:
            split_astr[i] = "#" + split_astr[i]
            newastr = "\n".join(split_astr)
            r = te.loada(newastr)
            if not is_oscillator(r):
                # If removing the reaction broke the oscillator, put it back
                split_astr[i] = split_astr[i][1:]
            else:
                # If the model is still an oscillator, keep the comment and count the removed reaction
                reactions_pruned += 1
    return reactions_pruned, "\n".join(split_astr)


def prune_models(path):
    """
    Remove unnecessary reactions from antimony models in a directory
    :param path: (str) Path to a directory containing antimony files
    :return: None
    """
    total_reactions_removed = 0
    total_models_evaluated = 0
    for file in os.listdir(path):
        if file.endswith(".ant"):
            total_models_evaluated += 1
            with open(os.path.join(path, file), "r") as f:
                astr = f.read()
            reactions_pruned, new_astr = prune_antimony_model(astr)
            if reactions_pruned > 0:
                total_reactions_removed += 1
                with open(os.path.join(path, file), "w") as f:
                    f.write(new_astr)
    print(f"Removed {total_reactions_removed} reactions from {total_models_evaluated} models")
    print(f"Average reactions removed per model = {total_reactions_removed / total_models_evaluated}")
    return total_reactions_removed, total_models_evaluated
