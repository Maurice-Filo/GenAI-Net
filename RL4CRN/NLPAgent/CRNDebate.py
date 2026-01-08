from typing import List, Dict, Any, Tuple
import json
import torch
from RL4CRN.environments.environment import Environment
from RL4CRN.NLPAgent.nlp_core import DebateGraph, NLPAgent # <-- Import your core classes

class CRNDebate(DebateGraph):
    """
    Abstract Base Class for any debate regarding Chemical Reaction Networks.
    Provides the infrastructure for:
    - Tracking the Hall of Fame (Best CRNs found).
    - Tracking Feedback (Recent attempts and their errors).
    - Simulating and Evaluating candidates proposed by the agents.
    """
    def __init__(self, project_id: str, location: str = "global", track_top_k: int = 50, feedback_history: int = 15):
        super().__init__(project_id, location)
        
        # --- Shared Memory Systems ---
        self.gemini_top_k = [] 
        self.track_top_k_limit = track_top_k
        self.feedback_buffer = [] 
        self.feedback_history_size = feedback_history

    def setup_agents(self):
        """
        Override this method to initialize NLPAgents, 
        add them to the graph, and define the listening edges.
        """
        raise NotImplementedError("Subclasses must implement agent setup.")

    def run_session(self, **kwargs):
        """
        Override this to define how a debate round is triggered.
        """
        raise NotImplementedError("Subclasses must implement the session run logic.")

    # =========================================================================
    #  SHARED DOMAIN LOGIC (Simulation, Evaluation, Memory)
    # =========================================================================

    def evaluate_and_transplant(self,
                                candidates: List[Dict],
                                crn_template,
                                max_added_reactions: int,
                                library,
                                stepper,
                                actuator,
                                compute_reward_func,
                                is_ordered_policy: bool,
                                logger=None) -> List[Any]:
        """
        Takes raw JSON candidates from the debate, simulates them in the RL environment,
        computes rewards, and updates the internal memory (HoF/Feedback).
        """
        valid_envs = []
        print(f"[Debate] Simulating {len(candidates)} candidates...")

        for cand in candidates:
            feedback_entry = {"reasoning": cand.get("reasoning", "No reasoning"), "reward": "N/A", "valid": False}

            try:
                r_ids = cand.get('reaction_ids', [])
                p_vals = cand.get('parameter_values', [])
                
                # --- Safety Check 1: Length Mismatch ---
                if len(r_ids) != max_added_reactions:
                    print(f"  -> [Skip] Length Mismatch: Got {len(r_ids)}, Expected {max_added_reactions}")
                    feedback_entry["reasoning"] += f" (Length Mismatch: {len(r_ids)} vs {max_added_reactions})"
                    self._update_feedback(feedback_entry)
                    continue
                
                if not r_ids or len(r_ids) != len(p_vals):
                    feedback_entry["reasoning"] += " (IDs/Params length mismatch)"
                    self._update_feedback(feedback_entry)
                    continue

                # --- Sorting Logic ---
                if is_ordered_policy:
                    paired = sorted(zip(r_ids, p_vals), key=lambda x: x[0])
                else:
                    paired = list(zip(r_ids, p_vals))
                
                sorted_ids, sorted_params = zip(*paired) if paired else ([], [])
                
                # --- Simulation ---
                gemini_env = Environment(crn_template, max_added_reactions, logger=logger)
                gemini_env.reset()
                
                trajectory_valid = True
                for r_idx, params in zip(sorted_ids, sorted_params):
                    if r_idx >= len(library): 
                        trajectory_valid = False
                        break
                    
                    # Sanitize Parameters (prevent <= 0 crash in LogNormal)
                    raw_params_list = list(params) if isinstance(params, (list, tuple)) else [params]
                    sanitized_params = [max(1e-6, float(p)) for p in raw_params_list]
                    
                    raw_action_dict = {
                        'reaction index': int(r_idx), 
                        'parameters': sanitized_params, 
                        'continuous parameters': sanitized_params, 
                        'discrete parameters': None 
                    }
                    
                    action_object = actuator.actuate(raw_action_dict)
                    gemini_env.step(action=action_object, stepper=stepper, raw_action=raw_action_dict)

                if not trajectory_valid: 
                    feedback_entry["reasoning"] += " (Invalid Reaction Index)"
                    self._update_feedback(feedback_entry)
                    continue

                if len(gemini_env.raw_actions_taken) != max_added_reactions:
                     print("  -> [Skip] Trajectory length mismatch after Sim")
                     continue

                # --- Reward Calculation ---
                reward_result = compute_reward_func(gemini_env.state)
                
                if isinstance(reward_result, (tuple, list)): raw_val = reward_result[0]
                else: raw_val = reward_result
                if torch.is_tensor(raw_val): r_float = raw_val.item()
                else: r_float = float(raw_val)

                if not hasattr(gemini_env.state, 'last_task_info'): gemini_env.state.last_task_info = {}
                gemini_env.state.last_task_info['reward'] = r_float
                
                # --- Success: Store Data ---
                valid_envs.append(gemini_env)
                self._update_internal_hof(r_float, gemini_env, cand)
                
                feedback_entry["reward"] = f"{r_float:.6f}"
                feedback_entry["valid"] = True
                feedback_entry["CRN"] = str(gemini_env.state)
    
                self._update_feedback(feedback_entry)
                print(f"  -> Valid. Loss: {r_float:.6f}")

            except Exception as e:
                print(f"  -> Candidate Failed: {e}")
                feedback_entry["reasoning"] += f" (Exception: {str(e)})"
                self._update_feedback(feedback_entry)
        
        return valid_envs

    def _format_hof(self, hof_iter):
        """Simplifies HoF objects to strings for the Prompt"""
        formatted = ""
        # Access list safely
        hof_list = list(hof_iter)
        for i, env in enumerate(hof_list[:3]):
            r = env.state.last_task_info.get('reward', 'N/A')
            formatted += f"Design #{i} | Loss: {r} | Structure: {env.state}\n"
        return formatted

    def _format_feedback(self):
        if not self.feedback_buffer: return "No feedback yet."
        output = ""
        for item in self.feedback_buffer[-3:]:
             output += f"Attempt: {item['reasoning']} -> Reward: {item['reward']} (Valid: {item['valid']})\n"
        return output

    def _update_feedback(self, entry):
        self.feedback_buffer.append(entry)
        if len(self.feedback_buffer) > self.feedback_history_size: 
            self.feedback_buffer.pop(0)

    def _update_internal_hof(self, reward, env, cand):
        entry = {"reward": reward, "crn_string": str(env.state), "original_candidate": cand}
        self.gemini_top_k.append(entry)
        # Assuming lower loss is better; if reward is higher-is-better, remove reverse
        self.gemini_top_k.sort(key=lambda x: x['reward']) 
        if len(self.gemini_top_k) > self.track_top_k_limit: 
            self.gemini_top_k = self.gemini_top_k[:self.track_top_k_limit]

    def get_analysis_data(self):
        return self.gemini_top_k