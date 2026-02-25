r"""
Neural-network policies for adding reactions to an IOCRN.

This module contains policy networks that map a *tensorized IOCRN observation* to a
distribution over **actions** that extend the CRN by adding one reaction.

In the default “add reaction by index” formulation, an action is a dictionary:


- `reaction index` (int):, which library reaction to add next
- `continuous parameters` (list[float]): sampled continuous parameters (masked per reaction)
- `discrete parameters` (list[int]): | None,
- `parameters` (array-like): concatenation of continuous+discrete (if any)

    
The policy factorizes the joint action distribution into a *structure* term and
(optional) *parameter* terms:

$$\pi(a | s) = \pi_{struct}(r | s) \cdot \pi_{cont}(\theta_c | s, r) \cdot \pi_{disc}(\theta_d | s, r, \theta_c)$$

Log-probabilities and entropies returned by the policy correspond to this factorization:

$$\log \pi(a|s) = \log \pi_{struct}(r|s) + \log \pi_{cont}(\theta_c|s,r) + \log \pi_{disc}(\theta_d|s,r,\theta_c)$$

$$H(\pi)       = w_s H(\pi_{struct}) + w_c H(\pi_{cont}) + w_d H(\pi_{disc}) \quad  \text{(weighted per head)}$$

Masking is used to:
- forbid selecting reactions already present in the IOCRN (structure logits masked to -∞),
- forbid sampling parameters that do not exist for the chosen reaction (dimension masks),
- forbid invalid discrete-category combinations when using a flattened logit space (logit masks).

Temperature scaling can be applied to the structure logits to control exploration:

$$
    \pi_{struct}(r|s) = \text{softmax}\left(\frac{z_r(s)}{T}\right)
$$

where T may be adapted online to target a desired entropy ratio.
"""

import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot
from RL4CRN.policies.parameter_generator_from_distribution import ParameterGeneratorFromDistribution
import numpy as np

class AddReactionByIndex(torch.nn.Module):
    r"""
    Policy network that samples *one reaction addition* for each element of a batch of IOCRNs.

    The policy has an encoder + multiple “heads”:

    - **Encoder**: maps the observation vector `state` to a learned embedding `h`.
    - **Structure head**: produces logits over `M` library reactions, then samples a reaction index.
    - **Continuous parameter generator** (optional): samples continuous parameters for the chosen reaction.
    - **Discrete parameter generator** (optional): samples discrete parameters for the chosen reaction.

    The action distribution factorizes as:

    $$\pi(a|s) = \pi_{struct}(r|s) \cdot \pi_{cont}(\theta_c|s,r) \cdot \pi_{disc}(\theta_d|s,r,\theta_c)$$

    where:

    - $r$ is the reaction index (0..M-1),
    - $\theta_c$ are continuous parameters (e.g. LogNormal),
    - $\theta_d$ are discrete parameters (e.g. Categorical).

    Notes:
    *State layout* (no input-influence observation):
        state ∈ R^{N×(M+K)}

    - state[:, :M]  : multi-hot “reactions present” indicator
    - state[:, M:]  : flattened parameter vector (0 where not present)

    If `allow_input_influence=True`, the expected state layout is larger
    (includes additional per-input parameter influence features). This path is
    partially scaffolded but not implemented end-to-end in the current code.

    Returns from `forward`:

    - sampled action dictionaries (unless `action` is provided),
    - log-probabilities (per batch element),
    - entropies (per batch element, weighted per head).
    """
    def __init__(self, num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, 
                 input_influence_head_attributes, masks=None, zero_reaction_idx=None, stop_flag=False,
                 continuous_distribution={"type": 'lognormal'}, 
                 discrete_distribution={"type": 'categorical', "categories": torch.tensor([1, 2])}, # TODO: generalize to different categories per dimension
                 entropy_weights_per_head=None,
                 structure_head_temperature={"target_entropy_ratio_to_max": 1.0, "initial_temperature": 1.0, "rate": 0.0, "current_temperature": 1.0},
                 allow_input_influence=False, device=None):
        """
        Initialize the AddReactionByIndex policy.

        Args:
            num_reactions : int
                Number of candidate reactions in the library (denoted M).
            num_parameters : int
                Size of the flattened global parameter vector across the library (denoted K).
                This corresponds to the “explicit” parameterization used by observers/tensorizers.
            num_inputs : int
                Number of IO inputs (denoted p). Only relevant when using input-influence features.
            encoder_attributes : dict
                Configuration for the encoder MLP (`hidden_size`, `num_layers`).
            deep_layer_size : int
                Dimensionality of the encoder output embedding h(s).
            structure_head_attributes : dict
                Configuration for the structure head MLP (`hidden_size`, `num_layers`).
            parameter_head_attributes : dict
                Configuration for parameter generator backbones (`hidden_size`, `num_layers`).
            input_influence_head_attributes : dict
                Reserved for a future input-influence head (currently not implemented).
            masks : dict or None
                Optional masks derived from the reaction library:
                - 'continuous': float mask of shape (M, max_num_continuous_params)
                - 'discrete'  : float mask of shape (M, max_num_discrete_params)
                - 'logit'     : bool mask of shape (M, total_num_discrete_combinations)
                These masks are used to ensure only existing parameters/logits are used for each reaction.
            zero_reaction_idx : int or None
                If provided, the policy will be allowed to resample the “zero reaction” more than once.
            stop_flag : bool
                If True, the policy will stop adding reactions when the “zero reaction” is selected.
            continuous_distribution : dict
                Continuous parameter distribution spec passed to ParameterGeneratorFromDistribution
                (e.g. {"type": "lognormal", ...}). The policy sets `dim` automatically from masks.
            discrete_distribution : dict
                Discrete parameter distribution spec (e.g. {"type": "categorical", "categories": ...}).
                The policy sets `dim` automatically from masks. Current implementation assumes
                the same categories for each discrete dimension.
            entropy_weights_per_head : dict or None
                Entropy weights for each head. Keys: {'structure','continuous','discrete','input_influence'}.
                Used to form a weighted entropy signal:
                    H_total = Σ_i w_i H_i
            structure_head_temperature : dict
                Temperature schedule state for the structure head. Expected keys:
                - target_entropy_ratio_to_max
                - initial_temperature
                - rate
                - current_temperature
                The logits are scaled as z/T before constructing the Categorical distribution.
            allow_input_influence : bool
                If True, the observation and architecture include additional features/heads for
                input influence. (Currently not implemented.)
            device : torch.device or None
                Device where parameters and tensors should live.

        Raises:
            NotImplementedError
                If `allow_input_influence=True` (input-influence head is not implemented).
        """

        super().__init__()

        # Record the IOCRN attributes
        self.M = num_reactions                                              # Total number of reactions
        self.K = num_parameters                                             # Total number of parameters (continuous + discrete) across all reactions
        self.p = num_inputs                                                 # Number of inputs in the IOCRN
        self.zero_reaction_idx = zero_reaction_idx                          # If provided, the index of the “zero reaction” in the library (allows resampling and/or stopping) 
        self.stop_flag = stop_flag                                          # If True, the policy will stop adding reactions when the “zero reaction” is selected (only relevant if zero_reaction_idx is provided)

        # Record the neural network attributes
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.structure_head_attributes = structure_head_attributes
        self.parameter_head_attributes = parameter_head_attributes
        self.input_influence_head_attributes = input_influence_head_attributes
        self.allow_input_influence = allow_input_influence
        self.device = device if device is not None else torch.device('cpu')
        self.structure_head_temperature = structure_head_temperature
        self.max_structure_entropy = np.log(self.M)  # Maximum entropy of the reaction structure distribution

        # Record the distribution attributes
        self.continuous_distribution = continuous_distribution
        self.discrete_distribution = discrete_distribution

        # Tensorize the masks, if provided, and tensorize them into the specified device
        if masks is None:
            masks = {'continuous': None, 'discrete': None, 'logit': None}
        self.continuous_parameter_mask = masks['continuous']
        self.continuous_parameter_mask = torch.tensor(self.continuous_parameter_mask, dtype=torch.float32).to(self.device) if self.continuous_parameter_mask is not None else None # Shape: (M, max_num_continuous_parameters)
        self.discrete_parameter_mask = masks['discrete']
        self.discrete_parameter_mask = torch.tensor(self.discrete_parameter_mask, dtype=torch.float32).to(self.device) if self.discrete_parameter_mask is not None else None  # Shape: (M, max_num_discrete_parameters)
        self.logit_mask = masks['logit']

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        if allow_input_influence:
            self.encoder = FFNN(input_size=self.M + (self.p + 1) * self.K, output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        else:
            self.encoder = FFNN(input_size=self.M + self.K, output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)

        # Define the reaction structure head that reads the deep layer representation to output the logits for the reaction structure
        self.reaction_structure_head = FFNN(input_size=deep_layer_size, output_size=self.M, hidden_size=structure_head_attributes["hidden_size"], num_layers=structure_head_attributes["num_layers"]).to(device=device)
        
        # Define the continuous parameter head that reads the deep layer representation and the reaction structure to deliver to the continuous parameter generator
        if self.continuous_parameter_mask is not None:

            # Set the dimension of the continuous distribution to the maximum number of continuous parameters across all reactions in the library
            self.continuous_distribution["dim"] = self.continuous_parameter_mask.shape[1] 

            # Construct the continuous parameter generator
            self.continuous_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.continuous_distribution, 
                                                                                     backbone_attributes={  "input_size": deep_layer_size + self.M, 
                                                                                                            "hidden_size": parameter_head_attributes["hidden_size"], 
                                                                                                            "num_layers": parameter_head_attributes["num_layers"]
                                                                                                            }, 
                                                                                    device=self.device
                                                                                    ).to(device=self.device)   
        else:
            self.continuous_parameter_generator = None

        # Define the discrete parameter head that reads the deep layer representation, the reaction structure, and the continuous parameters to deliver to the discrete parameter generator
        if self.discrete_parameter_mask is not None:
            
            # Set the dimension of the discrete distribution to the maximum number of discrete parameters across all reactions in the library
            self.discrete_distribution["dim"] = self.discrete_parameter_mask.shape[1]

            # Make the discrete distribution categories the same for all discrete parameters #TODO Allow different categories for different discrete parameters
            self.discrete_distribution["categories"] = self.discrete_distribution["dim"] * self.discrete_distribution["categories"] # List of tensors, each of shape (num_categories,) where num_categories is the number of categories for each discrete parameter

            # Calculate the input size of the discrete parameter head
            discrete_parameter_head_input_size = deep_layer_size + self.M + (self.continuous_parameter_mask.shape[1] if self.continuous_parameter_mask is not None else 0)

            # Construct the discrete parameter generator
            self.discrete_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.discrete_distribution, 
                                                                                     backbone_attributes={  "input_size": discrete_parameter_head_input_size, 
                                                                                                            "hidden_size": parameter_head_attributes["hidden_size"], 
                                                                                                            "num_layers": parameter_head_attributes["num_layers"]
                                                                                                            }, 
                                                                                    device=self.device
                                                                                    ).to(device=self.device)   
        else:
            self.discrete_parameter_generator = None
            
        # Define the input influence head that reads the deep layer representation, the reaction structure and the reaction rate to output the logits for the input influence
        if allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")
        else:
            self.input_influence_head = None

        # Record entropy weights per head
        self.entropy_weights_per_head = entropy_weights_per_head if entropy_weights_per_head is not None else {'structure': 1, 'continuous': 1, 'discrete': 1, 'input_influence': 1}
                  
    def forward(self, state, mode='full', action=None, structure_temp=None): 
        """
        Sample actions (or score provided actions) for a batch of IOCRN observations.

        Args:
            state : torch.Tensor
                Batched observation tensor of shape (N, D).
                For `allow_input_influence=False`, D = M + K and the layout is:

                - `state[:, :M]` : multi-hot indicator of reactions already present
                - `state[:, M:]` : flattened parameters (0 for absent reactions)
                The method asserts that the input contains no NaNs.
            mode : {"full", "partial"}

                - `full`: sample reaction structure and parameters.
                - `partial`: intended for parameter-only decisions given a fixed structure
                (not implemented in current code).
            action : list[dict] or None
                If provided, the policy **does not sample**; it computes log π(action|state)
                for the given batch of actions (used e.g. in SIL replay / scoring).
                Each dict must include at least:

                - `reaction index`
                - `continuous parameters` (if continuous generator exists)
                - `discrete parameters` (if discrete generator exists)
            structure_temp : float or None
                If provided, overrides the structure-head temperature `T` used for this call.

        Returns:
        - If `action is None`:
            - `actions` : list[dict]
                Sampled actions, one per batch element.
            - `log_probabilities` : torch.Tensor
                Log-probabilities per batch element, shape (N,).
                Computed as the sum of head log-probabilities:
                    log π(a|s) = log π_struct + log π_cont + log π_disc (+ log π_input_influence)
            - `entropies` : torch.Tensor
                Weighted entropy per batch element, shape (N,):
                    H = w_s H_struct + w_c H_cont + w_d H_disc
        - If `action is not None`:
            - `log_probabilities` : torch.Tensor
                Log-probabilities of the provided actions, shape (N,).

        Implementation details:
            **Structure sampling with masking**
            Let z(s) be the structure logits (N×M). Reactions already present are masked:
                z_masked = z(s) with z_masked[r_present] = -∞
            Then temperature scaling is applied:
                z_T = z_masked / T
            and a Categorical distribution is formed:
                r ~ Categorical(logits=z_T)

            **Adaptive temperature (training only)**
                When sampling (action is None) in training mode, the current temperature is nudged
                based on the observed mean structure entropy relative to the maximum entropy log(M).

            **Parameter generation**
                Continuous and discrete parameters are generated conditionally using
                `ParameterGeneratorFromDistribution`, and are masked so that nonexistent parameters
                are zeroed out and/or omitted from the returned per-sample lists.
        """

        # Validate the input has no NaNs
        assert state.isnan().sum() == 0, "Input contains NaN values."

        # Encode the observation
        encoded = self.encoder(state) # shape: (N, deep_layer_size)
        
        # Run the reaction structure head to generate the structure of the next reaction while masking out already existing reactions in the IOCRN 
        entropies = 0
        log_probabilities = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded) # shape: (N, M)

            # Mask out already existing reactions in the IOCRN
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(state[:,:self.M].bool(), float('-inf')) # shape: (N, M)

            # Unmask the zero reaction if zero_reaction_idx is provided, allowing it to be resampled multiple times
            if self.zero_reaction_idx is not None:
                masked_reaction_structure_logits[:, self.zero_reaction_idx] = reaction_structure_logits[:, self.zero_reaction_idx]

            # Mask out all reactions except the zero reaction if stop_flag is True and the zero reaction is present in the IOCRN, forcing the policy to select the zero reaction and stop adding meaningful reactions
            if self.stop_flag and self.zero_reaction_idx is not None:
                rows = (state[:, self.zero_reaction_idx] == 1) # shape: (N,), bool tensor 
                masked_reaction_structure_logits[rows, :] = float('-inf')
                masked_reaction_structure_logits[rows, self.zero_reaction_idx] = reaction_structure_logits[rows, self.zero_reaction_idx]

            # Apply temperature to the logits
            if structure_temp is not None:
                self.structure_head_temperature["current_temperature"] = structure_temp
            masked_reaction_structure_logits = masked_reaction_structure_logits / self.structure_head_temperature["current_temperature"]

            # Construct the categorical distribution over the library reactions and compute their entropies
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits) # batch of N categorical distributions, each over M categories
            structure_entropies = reaction_structure_distribution.entropy() # shape: (N,)
            entropies = self.entropy_weights_per_head['structure'] * structure_entropies # shape: (N,)

            # Adapt the temperature based on the entropy of the distribution
            if self.training and action is None:
                with torch.no_grad():
                    mean_structure_entropy = structure_entropies.mean().item()
                    if mean_structure_entropy < self.max_structure_entropy * self.structure_head_temperature["target_entropy_ratio_to_max"]:
                        self.structure_head_temperature["current_temperature"] += self.structure_head_temperature["rate"]
                    else:
                        self.structure_head_temperature["current_temperature"] -= self.structure_head_temperature["rate"]
                    self.structure_head_temperature["current_temperature"] = max(0.05, min(20.0, self.structure_head_temperature["current_temperature"]))


            # Sample the reaction structure from the distribution and compute the log probabilities of the sampled reactions
            samples_reaction_idx = reaction_structure_distribution.sample() if action is None else torch.tensor([a['reaction index'] for a in action], requires_grad=False).to(self.device)  # shape: (N,)
            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx) # shape: (N,)

            # Create the one-hot encoding of the sampled reactions
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device) # shape: (N, M)
               
        # Create parameter masks corresponding to the sampled reactions
        continuous_parameter_mask_subset = self.continuous_parameter_mask[samples_reaction_idx] if self.continuous_parameter_mask is not None else None # Shape: (N, max_num_continuous_parameters) or None
        discrete_parameter_mask_subset = self.discrete_parameter_mask[samples_reaction_idx] if self.discrete_parameter_mask is not None else None  # Shape: (N, max_num_discrete_parameters) or None
        logit_mask_subset = self.logit_mask[samples_reaction_idx] if self.logit_mask is not None else None  # Shape: (N, total_num_categories_for_all_discrete_parameters) or None

        # Initialize the sampled parameters
        if action is None:
            samples_continuous_parameters = None
            samples_discrete_parameters = None
        else:
            # tensorize the provided action parameters and pad them with zeros to match the maximum number of parameters across all reactions
            samples_continuous_parameters = torch.tensor([a['continuous parameters'] + [0.0]*(continuous_parameter_mask_subset.shape[1]-len(a['continuous parameters'])) for a in action], requires_grad=False).to(self.device) if self.continuous_parameter_generator is not None else None # shape: (N, max_num_continuous_parameters) or None
            samples_discrete_parameters = torch.tensor([a['discrete parameters'] + [0]*(discrete_parameter_mask_subset.shape[1]-len(a['discrete parameters'])) for a in action], requires_grad=False).to(self.device) if self.discrete_parameter_generator is not None else None # shape: (N, max_num_discrete_parameters) or None
        
        # Concatenate the encoded IOCRN with the one-hot encoding of the sampled reaction structure to form the input to the continuous parameter generator
        x = torch.cat([encoded, samples_reaction_hot], dim=-1) # shape: (N, deep_layer_size + M)

        # Run the continuous and discrete parameter generators to generate the parameters of the sampled reactions
        parameter_types = ['continuous', 'discrete']
        for type in parameter_types:
            match type:
                case 'continuous':
                    # Skip if there are no continuous parameters to generate
                    if self.continuous_parameter_generator is None:
                        continue

                    # Generate the continuous parameters samples, their log probabilities and entropies of their respective distributions
                    samples_continuous_parameters, log_probs_continuous_parameters, entropies_continuous_parameters = self.continuous_parameter_generator(x, mask=continuous_parameter_mask_subset, samples=samples_continuous_parameters) # shapes: (N, max_num_continuous_parameters), (N,), (N,)

                    # Accumulate the log probabilities and entropies
                    entropies = entropies + self.entropy_weights_per_head['continuous'] * entropies_continuous_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_continuous_parameters # shape: (N,)

                    # Mask out the parameters that do not exist for the sampled reactions
                    if action is None:
                        samples_continuous_parameters = samples_continuous_parameters * continuous_parameter_mask_subset if continuous_parameter_mask_subset is not None else samples_continuous_parameters

                    # Concatenate the encoded IOCRN, the one-hot encoding of the sampled reaction structure and the sampled continuous parameters to form the input to the discrete parameter generator
                    x = torch.cat([x, samples_continuous_parameters], dim=-1)

                case 'discrete':
                    # Skip if there are no discrete parameters to generate
                    if self.discrete_parameter_generator is None:
                        continue

                    # Generate the discrete parameters samples, their log probabilities and entropies of their respective distributions
                    samples_discrete_parameters, log_probs_discrete_parameters, entropies_discrete_parameters = self.discrete_parameter_generator(x, logit_mask=logit_mask_subset, dimension_mask=discrete_parameter_mask_subset, samples=samples_discrete_parameters) # shapes: (N, max_num_discrete_parameters), (N,), (N,)
                    
                    # Accumulate the log probabilities and entropies
                    entropies = entropies + self.entropy_weights_per_head['discrete'] * entropies_discrete_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_discrete_parameters # shape: (N,)
                    
                    # Mask out the parameters that do not exist for the sampled reactions
                    samples_discrete_parameters = samples_discrete_parameters * discrete_parameter_mask_subset if discrete_parameter_mask_subset is not None else samples_discrete_parameters
                    
                    # Concatenate the encoded IOCRN, the one-hot encoding of the sampled reaction structure, the sampled continuous parameters and the sampled discrete parameters to form the input to the input influence head, if applicable
                    x = torch.cat([x, samples_discrete_parameters], dim=-1)

        # Run the input influence head, if applicable, to generate the input influence 
        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")

        # If action is provided, return only the log probabilities (used for computing the probability of an external action)
        if action is not None:
            return log_probabilities
        
        # Otherwise, return the sampled actions, their log probabilities and entropies
        # Process the sampled parameters to return only the parameters that exist for the sampled reactions #TODO: This part should belong to the actuator not the agent as it is environment-specific? 
        if samples_continuous_parameters is not None:
            if continuous_parameter_mask_subset is not None:
                samples_continuous_parameters = [samples_continuous_parameters[i, continuous_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_continuous_parameters.shape[0])] # N-List of lists of continuous parameters, each sublist containing only the continuous parameters that exist for the sampled reaction
            else:
                samples_continuous_parameters = samples_continuous_parameters.cpu().numpy().tolist() # N-List of lists of continuous parameters, each sublist containing all the continuous parameters (no masking)
            
        if samples_discrete_parameters is not None:
            if discrete_parameter_mask_subset is not None:
                samples_discrete_parameters = [samples_discrete_parameters[i, discrete_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_discrete_parameters.shape[0])] # N-List of lists of discrete parameters, each sublist containing only the discrete parameters that exist for the sampled reaction
            else:
                samples_discrete_parameters = samples_discrete_parameters.cpu().numpy().tolist() # N-List of lists of discrete parameters, each sublist containing all the discrete parameters (no masking)

        # Collect the policy output, convert to numpy and move to CPU 
        samples_reaction_idx = samples_reaction_idx.cpu().numpy()

        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")
            
        else:
            if mode == 'full': # structure and parameters
                if samples_discrete_parameters is None:
                    action = [
                        {
                            'reaction index': r_idx,
                            'parameters': param_continuous,
                            'continuous parameters': param_continuous, 
                            'discrete parameters': None
                        }
                        for r_idx, param_continuous in zip(samples_reaction_idx, samples_continuous_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index and continuous parameters
                else:
                    action = [
                        {
                            'reaction index': r_idx,
                            'parameters': np.concatenate([param_continuous, param_discrete]), 
                            'continuous parameters': param_continuous, 
                            'discrete parameters': param_discrete
                        }
                        for r_idx, param_continuous, param_discrete in zip(samples_reaction_idx, samples_continuous_parameters, samples_discrete_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index, continuous and discrete parameters

            elif mode == 'partial': # rates
                raise NotImplementedError("The 'partial' mode is not implemented yet.")

            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
        # Return the samples, log probability and entropy
        return action, log_probabilities, entropies