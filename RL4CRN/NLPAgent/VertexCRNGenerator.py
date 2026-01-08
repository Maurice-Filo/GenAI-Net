import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import torch
from typing import List, Dict, Optional, Any, Tuple
from RL4CRN.environments.environment import Environment

class VertexCRNGenerator:
    def __init__(self, 
                 project_id: str, 
                 location: str = "europe-west1", 
                 model_name: str = "gemini-2.5-flash",
                 default_system_prompt: Optional[str] = None,
                 track_top_k: int = 50,
                 feedback_history_size: int = 10): # Number of past attempts to remember
        
        self.project_id = project_id
        self.location = location
        
        # Auth & Model
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        
        # Long-term Memory (Best ever)
        self.gemini_top_k = [] 
        self.track_top_k_limit = track_top_k
        
        # Short-term Memory (Recent Feedback)
        self.feedback_buffer = [] 
        self.feedback_history_size = feedback_history_size

        if default_system_prompt:
            self.system_prompt = default_system_prompt
        else:
            self.system_prompt = """
            You are an expert Synthetic Biologist and Control Theorist.
            You learn from feedback. You analyze previous attempts to see what worked and what failed.
            Output must be strictly valid JSON. You will be given some examples of CRNs that worked the best so far.
            You will also be given feedback on your previous suggestions. Act to minimize the loss reported in the feedback.
            """

    def generate_candidates(self, 
                          hall_of_fame_iter, 
                          task_description: str, 
                          reaction_library, 
                          num_candidates: int = 5) -> List[Dict]:
        
        # 1. Format Hall of Fame (The "Best so far")
        hof_context = ""
        top_examples = list(hall_of_fame_iter)[:5] 
        for i, env in enumerate(top_examples):
            r = env.state.last_task_info.get('reward', 'N/A')
            hof_context += f"--- HoF Example {i+1} (Loss: {r}) ---\n{env.state}\n"

        # 2. Format Feedback (The "Recent Attempts")
        feedback_context = ""
        if not self.feedback_buffer:
            feedback_context = "No previous feedback available (First iteration)."
        else:
            feedback_context = "Here are the results of your MOST RECENT suggestions:\n"
            for item in self.feedback_buffer:
                # We show the reasoning and the result
                status = "SUCCESS" if item['valid'] else "FAILED/INVALID"
                feedback_context += (
                    f"- Suggestion: {item['reasoning']}\n"
                    f"  Result: {status} | Loss: {item['reward']}\n"
                )
            feedback_context += "\nAnalyze this feedback. If a strategy resulted in high loss, try a different approach."

        # 3. Construct Library Menu
        library_menu = "AVAILABLE REACTIONS (Select by ID):\n"
        for idx, rxn in enumerate(reaction_library.reactions):
            library_menu += f"ID {idx}: {str(rxn)}\n"

        # 4. Construct Prompt
        full_prompt = f"""
        {self.system_prompt}

        === 1. THE TASK ===
        {task_description}

        === 2. FEEDBACK ON YOUR PREVIOUS ATTEMPTS ===
        {feedback_context}

        === 3. HALL OF FAME (Current Best Designs) ===
        {hof_context}

        === 4. CONSTRAINTS (Library) ===
        Select reactions ONLY from this list:
        {library_menu}

        === 5. GENERATION GOAL ===
        Generate {num_candidates} NEW, DISTINCT Chemical Reaction Networks.
        - Learn from the Feedback section above.
        - Combine motifs from the Hall of Fame.
        
        === 6. OUTPUT FORMAT ===
        Return a JSON Object with a list called 'candidates'.
        Format:
        {{
          "candidates": [
            {{
              "reasoning": "Based on the feedback that integral feedback failed, I am trying an incoherent feedforward loop.",
              "reaction_ids": [0, 5],
              "parameter_values": [[1.0], [0.5]] 
            }}
          ]
        }}
        """

        config = GenerationConfig(temperature=0.9, response_mime_type="application/json")

        try:
            response = self.model.generate_content(full_prompt, generation_config=config)
            data = json.loads(response.text)
            return data.get("candidates", [])
        except Exception as e:
            print(f"[VertexCRNGenerator] Generation Error: {e}")
            return []

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
        
        valid_envs = []
        print(f"[Gemini] Evaluating {len(candidates)} candidates...")

        for cand in candidates:
            # Prepare feedback entry
            feedback_entry = {
                "reasoning": cand.get("reasoning", "No reasoning provided"),
                "reward": None,
                "valid": False
            }

            try:
                r_ids = cand.get('reaction_ids', [])
                p_vals = cand.get('parameter_values', [])
                
                # Check formatting
                if len(r_ids) != len(p_vals):
                    self._update_feedback(feedback_entry) # Records as invalid
                    continue

                # --- 1. Sort Logic ---
                if is_ordered_policy:
                    paired = sorted(zip(r_ids, p_vals), key=lambda x: x[0])
                else:
                    paired = list(zip(r_ids, p_vals))
                sorted_ids, sorted_params = zip(*paired) if paired else ([], [])
                
                # --- 2. Simulation ---
                gemini_env = Environment(crn_template, max_added_reactions, logger=logger)
                gemini_env.reset()
                
                trajectory_valid = True
                
                for r_idx, params in zip(sorted_ids, sorted_params):
                    if r_idx >= len(library): 
                        trajectory_valid = False; break
                    
                    # Convert params safely
                    current_params_list = list(params) if isinstance(params, (list, tuple)) else [params]
                    
                    raw_action_dict = {
                        'reaction index': int(r_idx),
                        'parameters': current_params_list,
                        'continuous parameters': current_params_list,
                        'discrete parameters': None 
                    }

                    try:
                        action_object = actuator.actuate(raw_action_dict)
                        gemini_env.step(action=action_object, stepper=stepper, raw_action=raw_action_dict)
                    except Exception:
                        trajectory_valid = False; break

                if not trajectory_valid:
                    self._update_feedback(feedback_entry) # Records as invalid
                    continue

                # --- 3. Reward Calculation ---
                reward_result = compute_reward_func(gemini_env.state)
                
                # Unpack Tuple if necessary
                if isinstance(reward_result, (tuple, list)):
                    raw_val = reward_result[0]
                else:
                    raw_val = reward_result

                if torch.is_tensor(raw_val):
                    r_float = raw_val.item()
                else:
                    r_float = float(raw_val)
                
                # Metadata
                if not hasattr(gemini_env.state, 'last_task_info'):
                    gemini_env.state.last_task_info = {}
                gemini_env.state.last_task_info['reward'] = r_float
                
                valid_envs.append(gemini_env)
                self._update_internal_hof(r_float, gemini_env, cand)
                
                # --- 4. RECORD FEEDBACK (Success) ---
                feedback_entry["reward"] = f"{r_float:.4f}"
                feedback_entry["valid"] = True
                self._update_feedback(feedback_entry)

                print(f"  -> Gemini Candidate Injected. Loss: {r_float:.4f}")

            except Exception as e:
                # Catch-all for crashes
                feedback_entry["reasoning"] += f" (Crashed: {str(e)})"
                self._update_feedback(feedback_entry)
                print(f"  -> Gemini Reconstruction failed: {e}")
        
        return valid_envs

    def _update_feedback(self, entry: Dict):
        """Adds result to short-term memory buffer"""
        self.feedback_buffer.append(entry)
        # Keep only the last N attempts
        if len(self.feedback_buffer) > self.feedback_history_size:
            self.feedback_buffer.pop(0)

    def _update_internal_hof(self, reward: float, env, candidate_dict: Dict):
        entry = {
            "reward": reward,
            "crn_string": str(env.state),
            "original_candidate": candidate_dict
        }
        self.gemini_top_k.append(entry)
        self.gemini_top_k.sort(key=lambda x: x['reward'], reverse=False) # the reward is a loss (lower is better) TODO: remember to clarify this
        if len(self.gemini_top_k) > self.track_top_k_limit:
            self.gemini_top_k = self.gemini_top_k[:self.track_top_k_limit]
            
    def get_analysis_data(self):
        return self.gemini_top_k