import torch
import math
import numpy as np
from torch.distributions import Categorical
from RL4CRN.policies.add_reaction_by_index import AddReactionByIndex
from RL4CRN.utils.utils import batch_multi_hot

def log_combinations(n, k):
    r"""
    Compute the logarithm of the binomial coefficient, log C(n, k), in a numerically stable way.

    This helper is used to build *combinatorial priors/biases* over remaining action choices.
    It supports tensor-valued inputs and returns `-inf` for invalid pairs (k < 0 or k > n),
    which is convenient when treating invalid combinations as impossible events.

    Args:
        n : torch.Tensor
            Number of items available (can be broadcasted).
        k : torch.Tensor
            Number of items to choose (can be broadcasted).

    Returns:
        torch.Tensor
            `log(C(n, k))` with the broadcasted shape of `n` and `k`.
            Entries corresponding to invalid (n, k) pairs are `-inf`.

    Notes:
        Uses the identity:
        
        $$\log C(n, k) = \log\Gamma(n+1) - \log\Gamma(k+1) - \log\Gamma(n-k+1)$$

    and clamps intermediate values to avoid NaNs when masking invalid inputs.
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
    """
    Extension of `AddReactionByIndex` that enforces an *ordered* reaction-selection scheme.

    The base policy samples a reaction index from the library (excluding already-present reactions)
    and then samples its parameters. This subclass adds two extra structural constraints:

    1) **Template-aware ordering**
       At the first call in an episode/batch, the current IOCRN reaction multi-hot vector is
       snapshotted as a *template* (`template_mask`). Only reactions added *after* this snapshot
       are considered "added by the agent". Ordering constraints are applied **only** to these
       added reactions, so template reactions do not affect the allowed index range.

    2) **Sequentiality constraint**
       Once the agent has added at least one reaction, subsequent reactions must have an index
       strictly greater than the maximum index among the agent-added reactions so far. Concretely:
           r_next > max(added_indices)
       This is enforced with either:

       - a **soft** penalty (finite `constraint_strength`), or
       - a **hard** mask (`constraint_strength = inf`), making violations impossible.

    Additionally, an optional **combinatorial bias** term can be added to the structure logits
    to shape the policy toward a uniform distribution over *unordered sets* of a target size
    (rather than uniform over ordered action sequences).

    Compared to the base class, the parameters heads/generators are unchanged; only the structure
    sampling logits are modified prior to constructing the categorical distribution.
    """

    def __init__(self, num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, 
                 input_influence_head_attributes, 
                 target_set_size, 
                 masks=None,
                 continuous_distribution={"type": 'lognormal'}, 
                 discrete_distribution={"type": 'categorical', "categories": torch.tensor([1, 2])},
                 entropy_weights_per_head=None,
                 structure_head_temperature={"target_entropy_ratio_to_max": 1.0, "initial_temperature": 1.0, "rate": 0.0, "current_temperature": 1.0},
                 allow_input_influence=False, device=None, combinatorial_bias_enabled=True, constraint_strength=float('inf')):
        """
        Initialize the ordered-index reaction-addition policy.

        All parameters from `AddReactionByIndex` are supported. Additional parameters:
        
        Args:
            target_set_size : int
                Desired total number of reactions in the final CRN (including template reactions).
                Used to compute the combinatorial prior so that, under an uninformative policy,
                the probability of arriving at a particular final *set* is approximately uniform:
                    P(set) ∝ 1 / C(M, K)
                where M is library size and K is `target_set_size`.

            combinatorial_bias_enabled : bool, default=True
                If True, adds a combinatorial bias term to the structure logits that accounts for how
                many completions remain if a given index is chosen next.

            constraint_strength : float, default=inf
                Strength of the ordering constraint.
                - If finite: applies a subtractive penalty to out-of-order logits (soft constraint).
                - If infinite: treats out-of-order choices as impossible (hard mask).

        Internal state:
            - `template_mask` (torch.Tensor or None):
                Snapshot of the initial reaction multi-hot vector for the current episode/batch.
                Shape (N, M). Set on the first `forward` call after `reset_template()`.

            - `library_indices` (torch.Tensor):
                Float tensor [0, 1, ..., M-1] used to compute max indices efficiently.
        """
        
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
        self.constraint_strength = constraint_strength

        # self._initialize_structure_head_bias()

    def reset_template(self):
        """
        Reset the internal template snapshot.

        Call this at the start of a new episode (or whenever the “template CRN” changes) so that
        the next call to `forward` captures the current reaction multi-hot vector as `template_mask`.

        Why this matters:
            The ordering constraint is designed to apply only to reactions *added by the agent*.
            Resetting the template ensures that pre-existing/template reactions do not influence
            the computed `max_added_index` and therefore do not restrict future choices.
        """
        self.template_mask = None

    def forward(self, state, mode='full', action=None, structure_temp=None):
        """
        Sample or score actions under ordered-index and combinatorial constraints.

        This method mirrors `AddReactionByIndex.forward` but modifies the structure logits
        before sampling/scoring the reaction index.

        Args:
            state : torch.Tensor
                Batched observation tensor (N, D). The first M entries must be the reaction multi-hot
                vector indicating reactions present in the current IOCRN.
            mode : {"full", "partial"}
                - "full": sample structure + parameters (supported).
                - "partial": not implemented.
            action : list[dict] or None
                If provided, the method computes log π(action|state) for the given actions instead of
                sampling. The action dictionaries must include a "reaction index" and parameter fields
                consistent with the configured generators (same as base class).
            structure_temp : float or None
                Optional temperature override for the structure head logits.

        Returns:
            - If `action is None`:
                - `actions` (list[dict]):
                    Sampled actions, one per batch element.
                - `log_probabilities` (torch.Tensor):
                    Log-probability per batch element, including structure + parameter terms.
                - `entropies` (torch.Tensor):
                    Weighted entropy per batch element (structure + parameter heads).
            - If `action is not None`:
                - `log_probabilities` (torch.Tensor):
                    Log-probability per batch element for the provided actions.

        Ordering logic: 
            1. **Template snapshot (first call only)**
                If `template_mask` is not set, store `state[:, :M]` as the template.

            2. **Determine agent-added reactions**
                `added_reactions_mask = (state[:,:M] - template_mask) > 0.5`
                and compute: `num_added_by_agent`, `total_existing_counts`

            3. **Sequentiality mask**
            Let `max_added_index` be the maximum library index among *added* reactions.
            If the agent has added at least one reaction, indices <= max_added_index are penalized
            or masked (depending on `constraint_strength`).

            4. **Combinatorial bias (optional)**
            A bias term is added to each candidate reaction index i representing the log-count of
            ways to complete the remaining set after choosing i, accounting for template-fixed items.
            Invalid completions yield `-inf` and are hard-masked.

            5. **Hard vs soft masks**
            Hard mask:
                - template reactions (cannot re-select fixed/template entries)
                - impossible completions from combinatorial bias (`-inf`)
            Soft mask:
                - out-of-order indices (sequentiality violations), optionally penalized

            6. **Emergency valve**
            If all logits become `-inf` for any batch element, the last index is set to 0 to avoid
            crashing the categorical distribution construction.

            7. **Sampling / scoring**
            Build a Categorical over masked logits (with temperature) and sample or evaluate the
            provided indices.

            8. **Parameter generation**
            Delegates to the same continuous/discrete parameter generators as the base class.

        Notes:
            - This class does not change the parameterization; it only constrains structure sampling.
            - The “entropy correction” term when combinatorial bias is enabled modifies the structure
            entropy signal by adding E_p[bias], which corresponds to optimizing toward the biased prior
            (i.e., minimizing KL(p || exp(bias)) up to a constant).
        """

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
                reaction_structure_logits = torch.nan_to_num(reaction_structure_logits, nan=float('-inf'))

            # --- STEP 4: Combinatorial Bias ---
            reactions_left_to_pick = torch.clamp(self.target_set_size - total_existing_counts, min=0)
            k_req = (torch.clamp(reactions_left_to_pick, min=1) - 1).unsqueeze(-1)

            # Effective Availability (Memory Optimized)
            template_correction = torch.flip(torch.cumsum(torch.flip(self.template_mask, dims=[1]), dim=1), dims=[1])
            n_trailing = torch.flip(torch.arange(0, self.M, device=self.device), dims=[0]).unsqueeze(0)
            n_avail = n_trailing - template_correction

            combinatorial_bias = log_combinations(n_avail, k_req)
            
            is_done_mask = (reactions_left_to_pick <= 0).unsqueeze(-1)
            combinatorial_bias = combinatorial_bias.masked_fill(is_done_mask, 0.0)
            
            if self.combinatorial_bias_enabled:
                reaction_structure_logits = reaction_structure_logits + combinatorial_bias

            # --- STEP 5: Apply Masks (Split into Hard vs Soft) ---
            
            # A. HARD MASKS (Physical/Mathematical Impossibilities)
            # 1. Template: Cannot pick what is already fixed.
            # 2. Combinatorial -inf: Mathematically impossible to finish the set.
            hard_mask = self.template_mask.bool() | (combinatorial_bias == float('-inf'))
            
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(hard_mask, float('-inf'))

            # B. SOFT MASKS (Order Violations)
            # The sequentiality mask marks indices <= max_added (out of order).
            # We separate items that are already hard-masked to apply penalty only to physically valid but unordered items.
            soft_mask = sequentiality_mask.bool() & (~hard_mask)
            
            # Penalty strength: Turns the "Hard Wall" into a "Steep Hill" 
            soft_order_penalty = self.constraint_strength 
            
            # Apply Penalty
            # if finite
            if math.isfinite(soft_order_penalty):
                masked_reaction_structure_logits = masked_reaction_structure_logits - (soft_mask.float() * soft_order_penalty)
            else:
                # if infinite, treat as hard mask
                masked_reaction_structure_logits = masked_reaction_structure_logits.masked_fill(soft_mask, float('-inf'))

            
            # --- STEP 6: Emergency Valve ---
            all_logits_neg_inf = (masked_reaction_structure_logits == float('-inf')).all(dim=-1)
            if all_logits_neg_inf.any():
                masked_reaction_structure_logits[all_logits_neg_inf, -1] = 0.0

            # --- STEP 7: Sampling ---
            if structure_temp is not None:
                self.structure_head_temperature["current_temperature"] = structure_temp
            
            temp = max(1e-4, self.structure_head_temperature["current_temperature"])
            masked_reaction_structure_logits = masked_reaction_structure_logits / temp

            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits)
            structure_entropies = reaction_structure_distribution.entropy()
            
            # --- ENTROPY CORRECTION (Relative to Bias) ---
            if self.combinatorial_bias_enabled:
                # Subtract the combinatorial bias contribution from the entropy target.
                # This encourages the agent to match the combinatorial prior (uniform sets) 
                # rather than uniform actions when it is uncertain.
                
                # Zero out bias where probabilities are 0 (Hard Mask) to avoid NaN (0 * -inf)
                # Note: Soft masked items have finite prob and finite bias, so they contribute correctly.
                safe_bias = torch.where(hard_mask, torch.zeros_like(combinatorial_bias), combinatorial_bias)
                
                # Add Expectation[Bias] to Entropy
                # Target: Maximize H(P) + E_P[Bias]  <=> Minimize KL(P || exp(Bias))
                structure_entropies = structure_entropies + ((safe_bias * reaction_structure_distribution.probs).sum(dim=1))

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