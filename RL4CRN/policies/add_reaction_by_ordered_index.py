import torch
import math
import numpy as np
from torch.distributions import Categorical
from RL4CRN.policies.add_reaction_by_index import AddReactionByIndex
from RL4CRN.utils.utils import batch_multi_hot

def log_combinations(n, k):
    """
    Comupte log(C(n k)).

    Args:
    - n (torch.Tensor): number of items to choose from.
    - k (torch.Tensor): number of items to choose.

    Returns:
    - log_comb (torch.Tensor): log of the number of combinations C(n, k). 
    """
    valid_mask = (k >= 0) & (k <= n) # put -inf where invalid (as convenrtion)
    safe_n = torch.clamp(n, min=0.0)
    safe_k = torch.clamp(k, min=0.0)
    safe_k_for_sub = torch.min(safe_k, safe_n)
    
    log_comb = torch.lgamma(safe_n + 1) - \
               torch.lgamma(safe_k + 1) - \
               torch.lgamma(safe_n - safe_k_for_sub + 1)
    
    return log_comb.masked_fill(~valid_mask, float('-inf'))

class AddReactionByOrderedIndex(AddReactionByIndex):
    def __init__(self, num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, 
                 input_influence_head_attributes, 
                 target_set_size, 
                 masks=None,
                 continuous_distribution={"type": 'lognormal'}, 
                 discrete_distribution={"type": 'categorical', "categories": torch.tensor([1, 2])},
                 entropy_weights_per_head=None,
                 structure_head_temperature={"target_entropy_ratio_to_max": 1.0, "initial_temperature": 1.0, "rate": 0.0, "current_temperature": 1.0},
                 allow_input_influence=False, device=None, combinatorial_bias_enabled=True):
        
        super().__init__(num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, 
                 input_influence_head_attributes, masks,
                 continuous_distribution, 
                 discrete_distribution,
                 entropy_weights_per_head,
                 structure_head_temperature,
                 allow_input_influence, device)
        
        self.target_set_size = target_set_size  # Target number of reactions to reach [K] (this is needed so that P(action) = 1/C(M, K))
        
        # Internal state to track the "Template"
        self.template_mask = None 
        self.library_indices = torch.arange(self.M, device=self.device).float()
        self.combinatorial_bias_enabled = combinatorial_bias_enabled

        # self._initialize_structure_head_bias()

    def reset_template(self):
        """ 
        Call this at the start of a new episode/batch to reset the template snapshot. 
        This allows the agent to distinguish between template reactions and added reactions.
        """
        self.template_mask = None

    # def _initialize_structure_head_bias(self):
    #     """
    #     Initializes the final layer bias of the reaction_structure_head to a positive value.
    #     This counteracts the large negative Combinatorial Bias term at the start of training,
    #     preventing vanishing probabilities and exploding gradients.
    #     """
    #     try:
    #         # Access the reaction structure head (assumed to be an FFNN or Sequential)
    #         # We iterate to find the last Linear layer
    #         last_linear = None
    #         for module in self.reaction_structure_head.modules():
    #             if isinstance(module, torch.nn.Linear):
    #                 last_linear = module
            
    #         if last_linear is not None:
    #             # Calculate a heuristic positive bias.
    #             # The Combinatorial bias is roughly log(K/M).
    #             # We want Initial_Logit + log(K/M) ~ 0
    #             # So Initial_Logit ~ -log(K/M) = log(M/K)
    #             if self.target_set_size > 0:
    #                 heuristic_bias = math.log(max(1, self.M / self.target_set_size))
    #             else:
    #                 heuristic_bias = 1.0
                
    #             # Initialize the bias of the last layer
    #             if last_linear.bias is not None:
    #                 torch.nn.init.constant_(last_linear.bias, heuristic_bias)
    #                 print(f"Initialized Reaction Structure Head Bias to +{heuristic_bias:.2f}")
                
    #             # Optional: Initialize weights to be small to reduce random noise at start
    #             torch.nn.init.xavier_uniform_(last_linear.weight, gain=0.01)
                
    #     except Exception as e:
    #         print(f"Warning: Could not initialize structure head weights: {e}")

    def forward(self, state, mode='full', action=None, structure_temp=None):

        # --- STEP 1: Snapshot the Template (First Call Only) ---
        if self.template_mask is None:
            # We freeze the current state as the "Template"
            self.template_mask = state[:,:self.M].clone()
        
        # --- STEP 2: Identify "Added" Reactions ---
        # Current State - Template = What the agent added
        # Use > 0.5 for float tolerance (not necessary)
        added_reactions_mask = (state[:, :self.M] - self.template_mask) > 0.5
        
        num_added_by_agent = added_reactions_mask.sum(dim=1) # Shape (N,)
        total_existing_counts = state[:, :self.M].sum(dim=1) # Shape (N,)

        # --- STEP 3: Calculate Ordering Mask (Based on ADDED only) ---
        
        # A. Identify indices of reactions added by the agent (convert mask to indices)
        added_indices = added_reactions_mask.float() * self.library_indices.unsqueeze(0)
        
        # B. Find the Max Index among the ADDED reactions
        #    (Indices in the template are ignored here, which fixes your issue)
        max_added_index = torch.max(added_indices, dim=-1).values # Shape (N,)
        
        # C. Logic: If we have added reactions, we must pick > max_added_index
        mask_condition = self.library_indices.unsqueeze(0) <= max_added_index.unsqueeze(-1) # one means "not allowed"
        
        has_added_reactions = num_added_by_agent > 0
        
        # Only apply the mask if we have actually added something.
        # If added_reactions is empty, mask is all False (Order constraint hasn't started yet) (this is just a trick)
        sequentiality_mask = torch.where(has_added_reactions.unsqueeze(-1), mask_condition, torch.zeros_like(mask_condition, dtype=torch.bool))

        # encode the state 
        encoded = self.encoder(state) 
        
        entropies = 0
        log_probabilities = 0
        
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded)

            if torch.isnan(reaction_structure_logits).any():
                reaction_structure_logits = torch.nan_to_num(reaction_structure_logits, nan=float('-inf')) # remove NaNs, set to p=-inf # it was 0, check TODO

            # --- STEP 4: Combinatorial Bias ---
            # Bias based on remaining slots needed (clamping is just for numerical safety, probably not needed)
            reactions_left_to_pick = torch.clamp(self.target_set_size - total_existing_counts, min=0)
            k_req = (torch.clamp(reactions_left_to_pick, min=1) - 1).unsqueeze(-1)

            # this means: sum all trailing ones in the template mask, this will reduce the available choices
            template_correction = torch.flip(torch.cumsum(torch.flip(self.template_mask, dims=[1]), dim=1), dims=[1]) # we count also the current slot, but it's ok due to the masking later
            # count trailing empty slots
            n_trailing = torch.flip(torch.arange(0, self.M, device=self.device), dims=[0]).unsqueeze(0)
            n_avail = n_trailing - template_correction # how many available choices for the next pick if I choose this slot? (corrected for template)

            combinatorial_bias = log_combinations(n_avail, k_req) # weighting for all the choices 
            
            # Mute bias if done (this part is useless now, by how we use the policy)
            is_done_mask = (reactions_left_to_pick <= 0).unsqueeze(-1)
            combinatorial_bias = combinatorial_bias.masked_fill(is_done_mask, 0.0)
            
            if self.combinatorial_bias_enabled:
                reaction_structure_logits = reaction_structure_logits + combinatorial_bias # + as we work in log-prob space

            # --- STEP 5: Apply Masks ---
            # 1. The Template (Cannot pick what's already there)
            # 2. The Sequential Order (Strictly > last ADDED item)
            # 3. Impossible paths (Combinatorial -inf)
            full_mask = self.template_mask.bool() | sequentiality_mask.bool()
            full_mask = full_mask | (combinatorial_bias == float('-inf'))

            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(full_mask, float('-inf'))
            
            # --- STEP 6: Emergency Valve (NOTE: This should never happen!!) ---
            all_logits_neg_inf = (masked_reaction_structure_logits == float('-inf')).all(dim=-1)
            
            if all_logits_neg_inf.any():
                # Force last index as dummy action for broken/finished rows
                masked_reaction_structure_logits[all_logits_neg_inf, -1] = 0.0

            # --- STEP 7: Sampling ---
            if structure_temp is not None:
                self.structure_head_temperature["current_temperature"] = structure_temp
            
            temp = max(1e-4, self.structure_head_temperature["current_temperature"])
            masked_reaction_structure_logits = masked_reaction_structure_logits / temp

            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits)
            structure_entropies = reaction_structure_distribution.entropy()
            entropies = self.entropy_weights_per_head['structure'] * structure_entropies

            # Temp update logic
            if self.training and action is None:
                with torch.no_grad():
                    mean_structure_entropy = structure_entropies.mean().item()
                    if mean_structure_entropy < self.max_structure_entropy * self.structure_head_temperature["target_entropy_ratio_to_max"]:
                        self.structure_head_temperature["current_temperature"] += self.structure_head_temperature["rate"]
                    else:
                        self.structure_head_temperature["current_temperature"] -= self.structure_head_temperature["rate"]
                    self.structure_head_temperature["current_temperature"] = max(0.05, min(20.0, self.structure_head_temperature["current_temperature"]))

            samples_reaction_idx = reaction_structure_distribution.sample() if action is None else torch.tensor([a['reaction index'] for a in action], requires_grad=False).to(self.device)
            
            # Bound guard
            samples_reaction_idx = torch.clamp(samples_reaction_idx, 0, self.M - 1)

            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx)
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device)
            
        # back to parameter generation
        continuous_parameter_mask_subset = self.continuous_parameter_mask[samples_reaction_idx] if self.continuous_parameter_mask is not None else None
        discrete_parameter_mask_subset = self.discrete_parameter_mask[samples_reaction_idx] if self.discrete_parameter_mask is not None else None
        logit_mask_subset = self.logit_mask[samples_reaction_idx] if self.logit_mask is not None else None

        if action is None:
            samples_continuous_parameters = None
            samples_discrete_parameters = None
        else:
            samples_continuous_parameters = torch.tensor([a['continuous parameters'] + [0.0]*(continuous_parameter_mask_subset.shape[1]-len(a['continuous parameters'])) for a in action], requires_grad=False).to(self.device) if self.continuous_parameter_generator is not None else None
            samples_discrete_parameters = torch.tensor([a['discrete parameters'] + [0]*(discrete_parameter_mask_subset.shape[1]-len(a['discrete parameters'])) for a in action], requires_grad=False).to(self.device) if self.discrete_parameter_generator is not None else None
        
        x = torch.cat([encoded, samples_reaction_hot], dim=-1)

        # --- NaN Guard for Generator Input (Fix for std >= 0 crash) ---
        if torch.isnan(x).any() or torch.isinf(x).any():
             x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        parameter_types = ['continuous', 'discrete']
        for type in parameter_types:
            match type:
                case 'continuous':
                    if self.continuous_parameter_generator is None:
                        continue
                    samples_continuous_parameters, log_probs_continuous_parameters, entropies_continuous_parameters = self.continuous_parameter_generator(x, mask=continuous_parameter_mask_subset, samples=samples_continuous_parameters)
                    entropies = entropies + self.entropy_weights_per_head['continuous'] * entropies_continuous_parameters
                    log_probabilities = log_probabilities + log_probs_continuous_parameters
                    if action is None:
                        samples_continuous_parameters = samples_continuous_parameters * continuous_parameter_mask_subset if continuous_parameter_mask_subset is not None else samples_continuous_parameters
                    x = torch.cat([x, samples_continuous_parameters], dim=-1)

                case 'discrete':
                    if self.discrete_parameter_generator is None:
                        continue
                    samples_discrete_parameters, log_probs_discrete_parameters, entropies_discrete_parameters = self.discrete_parameter_generator(x, logit_mask=logit_mask_subset, dimension_mask=discrete_parameter_mask_subset, samples=samples_discrete_parameters)
                    entropies = entropies + self.entropy_weights_per_head['discrete'] * entropies_discrete_parameters
                    log_probabilities = log_probabilities + log_probs_discrete_parameters
                    samples_discrete_parameters = samples_discrete_parameters * discrete_parameter_mask_subset if discrete_parameter_mask_subset is not None else samples_discrete_parameters
                    x = torch.cat([x, samples_discrete_parameters], dim=-1)

        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")

        if action is not None:
            return log_probabilities
        
        if samples_continuous_parameters is not None:
            if continuous_parameter_mask_subset is not None:
                samples_continuous_parameters = [samples_continuous_parameters[i, continuous_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_continuous_parameters.shape[0])]
            else:
                samples_continuous_parameters = samples_continuous_parameters.cpu().numpy().tolist()
            
        if samples_discrete_parameters is not None:
            if discrete_parameter_mask_subset is not None:
                samples_discrete_parameters = [samples_discrete_parameters[i, discrete_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_discrete_parameters.shape[0])]
            else:
                samples_discrete_parameters = samples_discrete_parameters.cpu().numpy().tolist()

        samples_reaction_idx = samples_reaction_idx.cpu().numpy()

        if mode == 'full':
            if samples_discrete_parameters is None:
                action = [
                    {'reaction index': r_idx, 'parameters': param_continuous, 'continuous parameters': param_continuous, 'discrete parameters': None}
                    for r_idx, param_continuous in zip(samples_reaction_idx, samples_continuous_parameters)
                ]
            else:
                action = [
                    {'reaction index': r_idx, 'parameters': np.concatenate([param_continuous, param_discrete]), 'continuous parameters': param_continuous, 'discrete parameters': param_discrete}
                    for r_idx, param_continuous, param_discrete in zip(samples_reaction_idx, samples_continuous_parameters, samples_discrete_parameters)
                ]
        elif mode == 'partial':
             raise NotImplementedError("The 'partial' mode is not implemented yet.")
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return action, log_probabilities, entropies