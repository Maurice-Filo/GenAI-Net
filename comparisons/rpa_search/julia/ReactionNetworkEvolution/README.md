# ReactionNetworkEvolution.jl
## Purpose
This module allows the user to easily run an evolutionary algorithm to create mass-action chemical reaction networks with specific behaviors or to match time series data. It is intended as a research tool for computational systems biology.

There are three primary use cases for this packge:
1. Evolving reaction networks with specific behaviors. The default is oscillation, but other behaviors may be added in the future. Additionally, users can input time series that reflect those desired behaviors.
2. Evolving reaction networks (or ensemble models) to fit time series data. Often systems biologist will have timeseries data from experiments and will want to search for a reaction network (or group of them) that can recapitulate the time series data. This is useful for building models, informing future experiments or looking at output models for common reaction themes which may indicate the presence of those reactions in the ground truth.
3. Parameter (rate constants) fitting. In the case where model topology is known (for example, the biologist has already built the model from first principles) and wants to generate parameter values for those reactions that will recapitulate experimental time series data.


## How to use
### Installing Julia
This module uses the julia programming language. Installation instructions can be found [here](https://julialang.org/downloads/).

### Quick Start
1. **Download the package** <br>
   From a julia console:
   ```
   using Pkg
   Pkg.add("ReactionNetworkEvolution")
   ```
2. **Run evolution with default settings**
   Remaining in the julia console:
   ```
   import ReactionNetworkEvolution as rne
   rne.run_evolution()
   ```

   **NOTE:** If an output directory is not supplied, then one will  be created in the current working directory. Make sure that the working directory allows new directories to be made.
3. **Customize Evolution (optional)**
   The run_evolution() function takes several optional keyword arguments:
  * ```ntrials```: the number of trials to run (100 by default)
  * ```ngenerations```: the number of generations per batch (800 by default)
  * ```population_size```: the number of networks in a population
  * ```pathtosettings```: path to a json file storing additional custom settings
  * ```outputpath```: path to a directory where evolution output will be written
  * ```seed```: specify a seed for the random number generator
    
  For example, to change the population_size to 200 via the command line: <br>
  ```rne.run_evolution(population_size=200)```


## Additional Custom Settings
#### Specify Additional Settings in a JSON File
Additional settings can be specified in a JSON file and supplied to the run_evolution() function as a keyword argument. <br>
```run_evolution(pathtosettings="/home/name/path/to/your/settings.json")```

An example of a JSON file specifying custom settings: <br>
```
{"chemical_species_names": ["A", "B", "C"],
"initial_concentrations": [5.0, 3.0, 6.0],
"reaction_probabilities": [0.2, 0.3, 0.3, 0.2],
"ngenerations": 500,
"population_size": 200,
}
```
Any settings that are not specified in the JSON file will be set to the default value.

#### Description of Settings and Default Values
<table>
  <tr>
    <th>Setting Name</th>
    <th>Default Value</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>ngenerations</td>
    <td>800</td>
    <td>The number of generations</td>
  </tr>
  <tr>
    <td>population_size</td>
    <td>100</td>
    <td>The number of reaction networks in the population</td>
  </tr>
  <tr>
    <td>nreactions</td>
    <td> 5 </td>
    <td>The number of reactions in each network at the beginning of evolution (this number may be different in final networks)</td>
  </tr>
  <tr>
    <td>chemical_species_names</td>
    <td>["S0", "S1", "S2"]</td>
    <td>List of the chemical species names</td>
  </tr>
  <tr>
    <td>initial_concentrations</td>
    <td>[1.0, 5.0, 9.0]</td>
    <td>The initial concentrations for the chemical species in chemical_species_names</td>
  </tr>
  <tr>
    <td>seed</td>
    <td></td>
    <td>Random seed</td>
  </tr>
  <tr>
    <td>portion_elite</td>
    <td>0.1</td>
    <td>Portion of best networks to copy to the next generation</td>
  </tr>
  <tr>
    <td>portion_delete</td>
    <td>0.1</td>
    <td>Portion of worst networks to remove</td>
  </tr>
  <tr>
    <td>tournament_select</td>
    <td>false</td>
    <td>If true, use tournament selection to choose non-elite networks</td>
  </tr>
  <tr>
    <td>reaction_probabilities</td>
    <td>[0.1, 0.4, 0.4, 0.1]</td>
    <td>Probability of adding uni-uni, uni-bi, bi-uni, and bi-bi reaction types</td>
  </tr>
  <tr>
    <td>rateconstant_range</td>
    <td>[0.1, 50]</td>
    <td>Min and max values for rate constants in new reactions or when choosing a completely new rate constant (values can be mutated to outside this range for existing reactions)</td>
  </tr>
  <tr>
    <td>p_rateconstant_mutation</td>
    <td>0.6</td>
    <td>Probability of changing a rate constant (as opposed to adding/deleting a reaction)</td>
  </tr>
  <tr>
    <td>p_new_rateconstant</td>
    <td>0.15</td>
    <td>If mutating a rate constant, probability that a completely new rate constant will be selected from the rate constant range</td>
  </tr>
  <tr>
    <td>p_crossover</td>
    <td>0</td>
    <td>Probability of crossover</td>
  </tr>
  <tr>
    <td>same_fitness_crossover</td>
    <td>false</td>
    <td>If true, networks with similar fitness will be considered equally fit and crossed over more leniently</td>
  </tr>
  <tr>
    <td>same_fitness_percent_range</td>
    <td>0.05</td>
    <td>If using same fitness crossover, networks with fitnesses within 1-x and 1+x will be considered the same fitness </td>
  </tr>
  <tr>
    <td>p_mutation</td>
    <td>1</td>
    <td>Probability of mutating a network by modifying rate constants or adding/deleting reactions</td>
  </tr>
  <tr>
    <td>exclusive_crossover_mutation</td>
    <td>false</td>
    <td>If true, networks will EITHER be crossed over OR mutated but never both</td>
  </tr>
  <tr>
    <td>enable_speciation</td>
    <td>true</td>
    <td>If true, networks will be divided into species and compete amongst themselves</td>
  </tr>
  <tr>
    <td>starting_delta</td>
    <td>0.65</td>
    <td>The speciation threshold at the beginning of evolution</td>
  </tr>
  <tr>
    <td>delta_step</td>
    <td>0.1</td>
    <td>The amount by which delta is adjusted to achieve the target number of species</td>
  </tr>
  <tr>
    <td>target_num_species</td>
    <td>10</td>
    <td>The threshold to split reaction networks into species will be adjusted up and down to attempt to maintain this number of species. </td>
  </tr>
  <tr>
    <td>use_seed_network</td>
    <td>false</td>
    <td>If true, the first generation will be populated with copies of a seed network loaded from the specified path</td>
  </tr>
  <tr>
    <td>seed_network_path</td>
    <td></td>
    <td>If using a seed network, the path to the network</td>
  </tr>
  <tr>
    <td>randomize_seed_network_rates</td>
    <td>true</td>
    <td>If true and using a seed network, the reaction rate constants will be selected randomly from the rate constant range</td>
  </tr>
  <tr>
    <td>average_fitness</td>
    <td>false</td>
    <td>If true, each species fitness will be the average of its networks - otherwise species fitness is the top network’s fitness</td>
  </tr>
  <tr>
    <td>rateconstant_distance_weight</td>
    <td>0.0</td>
    <td>Weight given to different paramters values when calculating distance between two networks - if 0, parameters will not be considered when calculating distance</td>
  </tr>
  <tr>
    <td>objective_data_path</td>
    <td></td>
    <td>If generating networks to fit data, path to .csv file with data to fit - if no path is given, ReactionNetworkEvolution.jl will attempt to generate oscillators</td>
  </tr>
  <tr>
    <td>track_fitness</td>
    <td>true</td>
    <td>Track information about evolution such as top fitness, number of species, etc for each generation</td>
  </tr>
  <tr>
    <td>writeout_threshold</td>
    <td>0.05</td>
    <td>If the best network reaches this fitness or higher, evolution will be stopped</td>
  </tr>
  <tr>
    <td>process_output_oscillators</td>
    <td>true</td>
    <td>If true, automatically test if evolved networks are oscillators or not</td>
  </tr>
  
</table>

## Use Cases
### 1. Evolving reaction networks with specific behaviors
This package was primarily designed to generate mass-action chemical reaction networks with oscillatory behavior. With the defualt settings, it will generate oscillators with 3 chemical species.
1. **Download the package** <br>
   From a julia console:
   ```
   using Pkg
   Pkg.add("ReactionNetworkEvolution")
   ```
2. **Run evolution with default settings**
   Remaining in the julia console:
   ```
   import ReactionNetworkEvolution as rne
   rne.run_evolution()
   ```

### 2. Evolving reaction networks to fit time series data
1. Supply the time series data as a CSV file. 
2. Create a JSON file with at least the following setting to direct the package to the time series CSV and disable sorting oscillators

```
{"objective_data_path": "/path/to/timeseries_data.csv",
"process_output_oscillators": false}
```

3. Run evolution supplying the JSON file:

 ```
   import ReactionNetworkEvolution as rne
   rne.run_evolution(pathtosettings="/path/to/settingsfile.json")
 ```

### 3. Evolve paramters with fixed topology network
1. Supply the time series data as a CSV file
2. Supply the network topology as an antimony file. 
3. Create a JSON file with at least the following settings:

```
{"objective_data_path": "/path/to/timeseries_data.csv",
"process_output_oscillators": false,
"seed_network_path": "/path/to/antimonyfile.txt",
"rateconstant_distance_weight": 1.0,
"p_rateconstant_mutation" 1.0}
```
The rateconstant_distance_weight setting enables the package to consider differences in reaction weight constants when comparing twp networks. This is necessary if all networks have the same topology, although some adjustments to the value may be necessary.

Setting p_rateconstant_mutation to 1.0 prevents the algorithm from adding or deleting reactions during evolution.

4. Run evolution supplying the JSON file:

 ```
   import ReactionNetworkEvolution as rne
   rne.run_evolution(pathtosettings="/path/to/settingsfile.json")
 ```
