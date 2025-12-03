Notation:
- N: Batch size
- M: Number of reactions in the library
- n: Number of species in the IOCRN
- m: Number of reactions in the IOCRN (m is increased by one whenever a reaction is added to the IOCRN)
- T: Final number of reactions to be generated
- p: Number of inputs to the IOCRN
- q: Number of outputs of the IOCRN
- S_R: Stoichiometry coefficients matrix of the reactants, shape (n,m)
- S_P: Stoichiometry coefficients matrix of the products, shape (n,m)
- S: Stoichiometry matrix, shape (n,m)
- c: rate constants vector (m,)

Data Structures:
- states: A 3-tuple containing:
    - states[0] (indices of reactions): A numpy array (np.int64) representing the indices of the reactions in the IOCRNs batch. Shape: (N, m)
    - states[1] (rate constants): A numpy array (np.float64) representing the rate constants of the reactions in the IOCRNs batch. Shape: (N, m)
    - states[2] (input influences): A p-list of numpy arrays (np.int64). Each array in the list is associated with one input, and contains the indices of the reactions influence by this input. 
                 Shape (N, #), where # is the maximum number of reactions in any CRN in the batch influenced by this input.

- actions: An N-list of dictionaries representing the batch of sampled reactions. Each dictionary in the list contains:
    - 'reaction index': The index (np.int64) of the sampled reaction (if mode is 'full').
    - 'rate constant': The sampled reaction rate (np.float32).
    - 'input influence index': The index (np.int64) of the input influence (if allow_input_influence is True).


## Notes on pycuda installation

To install pycuda in a venv, you have to export the following envirnomental varaibles:

```{bash}
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# install with
python -m pip install --no-cache-dir --no-build-isolation pycuda
```