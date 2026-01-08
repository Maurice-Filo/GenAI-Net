import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import torch
import time
from typing import List, Dict, Optional, Any, Tuple
from RL4CRN.environments.environment import Environment

class VertexMultiAgentDebate:
    def __init__(self, 
                 project_id: str, 
                 location: str = "global", 
                 fast_model_name: str = "gemini-2.5-flash",        # For Opportunist, Contrarian, Skeptic
                 smart_model_name: str = "gemini-3-pro-preview",   # For Narrator, Player
                 track_top_k: int = 50,
                 feedback_history_size: int = 15):
        
        self.project_id = project_id
        self.location = location
        
        # --- Auth & Model Initialization ---
        vertexai.init(project=project_id, location=location)
        self.fast_model = GenerativeModel(fast_model_name)
        self.smart_model = GenerativeModel(smart_model_name)
        
        print(f"[VertexMultiAgentDebate] Initialized. Smart: {smart_model_name}, Fast: {fast_model_name}")

        # --- Memory Systems ---
        # Long-term Memory (Best ever candidates found by Gemini)
        self.gemini_top_k = [] 
        self.track_top_k_limit = track_top_k
        
        # Short-term Memory (Recent Feedback loop)
        self.feedback_buffer = [] 
        self.feedback_history_size = feedback_history_size

        # --- Personas ---
        self.personas = {
            "Narrator": """You are the NARRATOR. You are the only agent who sees the true rewards (losses). 
            Your goal: Summarize the current situation of the Hall of Fame objectively. 
            Do NOT suggest specific solutions yet. Just present the facts, the best known motifs, 
            and the relevant scientific literature context. Be concise and factual.""",

            "Opportunist": """You are the OPPORTUNIST. 
            Your goal: Look at the Hall of Fame examples provided by the Narrator. 
            Propose 5 strategies that do the MINIMUM change to these existing winners to preserve their function while optimizing cost. 
            "Copy smart, don't reinvent." Reuse existing motifs. Be efficient.""",

            "Contrarian": """You are the CONTRARIAN. 
            Your goal: Ignore the Hall of Fame. Assume the current best solutions are local minima (sub-optimal). 
            Propose 5 radically DIFFERENT, unique topologies. Challenge the assumptions made by the Opportunist. 
            Think outside the box. Be disruptive.""",

            "Skeptic": """You are the SKEPTIC (The Scientist). 
            Your goal: Ignore the game theory. Focus on the PHYSICS and CHEMISTRY. 
            Analyze the proposals so far based on Control Theory (Integrators, Feedforward loops, Band-pass filters). 
            Critique the other agents if their biology is unsound. Suggest rigorous corrections.""",

            "Player": """You are the PLAYER. 
            Your goal: Synthesize everything. Read the Narration, the Opportunist's hacks, the Contrarian's wild ideas, and the Skeptic's warnings. 
            Be very careful about the choices you pick and be toughtful about why they are the right ones. The success of the team depends on you.
            Use the suggestion and critiques as your base, but you are free to deviate if you have a better idea.
            Output exactly 10 FINAL PROPOSALS in strict JSON format. 
            Balance exploration (Contrarian) and exploitation (Opportunist). ensure the reaction IDs exist in the library."""
        }

    def _generate_agent_response(self, model, role: str, context: str, temperature: float = 0.7) -> str:
        """Helper to get a single turn of dialogue from a specific agent."""
        prompt = f"""
        [SYSTEM ROLE]
        {self.personas[role]}
        
        [CURRENT CONTEXT]
        {context}
        
        [INSTRUCTION]
        Provide your output for this round. Keep it concise (under 200 words).
        """
        try:
            response = model.generate_content(prompt, generation_config=GenerationConfig(temperature=temperature))
            return response.text.strip()
        except Exception as e:
            print(f"[VertexMultiAgentDebate] Error in {role} generation: {e}")
            return f"[Silenced due to error: {e}]"

    def run_debate_and_generate(self, 
                                hall_of_fame_iter, 
                                task_description: str, 
                                reaction_library,
                                iteration=None,
                                max_added_reactions=None) -> List[Dict]:
        """
        Orchestrates the multi-agent debate and returns the final JSON candidates.
        """
        
        # 1. Format Context Data
        hof_context = "--- HALL OF FAME (BEST SO FAR) ---\n"
        # We iterate the HoF object (assuming it's iterable or has __iter__)
        top_examples = list(hall_of_fame_iter)[:5] 
        for i, env in enumerate(top_examples):
            # Safe access to reward/loss
            r = env.state.last_task_info.get('reward', 'N/A')
            hof_context += f"Design #{i+1} | Loss: {r} | Structure: {env.state}\n"

        feedback_context = "--- RECENT FEEDBACK (Last 5 Attempts) ---\n"
        if self.feedback_buffer:
            for item in self.feedback_buffer[-5:]:
                try:
                    status = "VALID" if item['valid'] else "INVALID"
                    feedback_context += f"Attempt [{status}]\n"
                    feedback_context += f"  Reasoning: {item['reasoning']}\n"
                    feedback_context += f"  Loss: {item['reward']}\n"
                    feedback_context += f"  Structure: {item['CRN']}\n"
                    feedback_context += "\n"
                except KeyError:
                    print("[VertexMultiAgentDebate] Warning: Incomplete feedback entry encountered.")
        else:
            feedback_context += "No previous feedback available yet.\n"

        library_menu = "--- AVAILABLE REACTIONS ---\n"
        for idx, rxn in enumerate(reaction_library.reactions):
            library_menu += f"ID {idx}: {str(rxn)}\n"

        # ==========================================
        # PHASE 1: THE NARRATOR (Smart Model)
        # ==========================================
        print(f"\n[Debate] Narrator is analyzing the Hall of Fame...")
        # here I want to underline that the task is harder as it must be solved with fewer reactions and with a specific number of species
        # task_description_details = f"You must explicitly consider and state that the produced chemical reaction netwok uses EXACTLY {max_added_reactions} reactions from the library provided, which makes the task more challenging."
        narrator_input = f"TASK:\n{task_description}\n\nBEST RESULTS SO FAR\n{hof_context}\nRESULTS IN THE LAST ITERATION\n{feedback_context}\n"
        narration = self._generate_agent_response(self.smart_model, "Narrator", narrator_input, temperature=0.3)
        
        if iteration is not None:
            transcript = f"--- DEBATE TRANSCRIPT: ITERATION {iteration} ---\n\n"
        else:
            transcript = f"--- DEBATE TRANSCRIPT ---\n\n"
        transcript += f"NARRATOR:\n{narration}\n\n"

        # ==========================================
        # PHASE 2: THE SPECIALISTS (Fast Model)
        # ==========================================
        # Opportunist (Moderate Temp)
        print(f"[Debate] Opportunist is proposing exploitative strategies...")
        opportunist_input = f"TASK:\n{task_description}\n\n{transcript}\n\n[TASK] Propose 5 conservative modifications based on the Narrator's analysis."
        opportunist_msg = self._generate_agent_response(self.fast_model, "Opportunist", opportunist_input, temperature=0.5)
        transcript += f"OPPORTUNIST:\n{opportunist_msg}\n\n"

        # Contrarian (High Temp)
        print(f"[Debate] Contrarian is proposing radical alternatives...")
        contrarian_input = f"TASK:\n{task_description}\n\n{transcript}\n\n[TASK] Propose 5 radical alternatives ignoring the Narrator's safety."
        contrarian_msg = self._generate_agent_response(self.fast_model, "Contrarian", contrarian_input, temperature=0.9)
        transcript += f"CONTRARIAN:\n{contrarian_msg}\n\n"

        # Skeptic (Low Temp)
        print(f"[Debate] Skeptic is critiquing...")
        skeptic_input = f"TASK:\n{task_description}\n\n{transcript}\n\n[TASK] Critique the Opportunist and Contrarian scientifically."
        skeptic_msg = self._generate_agent_response(self.fast_model, "Skeptic", skeptic_input, temperature=0.2)
        transcript += f"SKEPTIC:\n{skeptic_msg}\n\n"

        # ==========================================
        # PHASE 3: THE PLAYER (Smart Model) - DECISION
        # ==========================================
        print(f"[Debate] Player is synthesizing final candidates...")
        
        final_prompt = f"""
        [SYSTEM ROLE]
        {self.personas['Player']}

        [MAXIMUM NUMBER OF REACTIONS]
        You must select stirctly N={max_added_reactions} reactions for the final designs.

        [DEBATE TRANSCRIPT]
        {transcript}

        [LIBRARY CONSTRAINT]
        Use ONLY these IDs from the reaction library:
        {library_menu}

        [FEEDBACK FROM PREVIOUS ITERATIONS]
        This is what you tries the last time and how it went, try to do better:
        {feedback_context}
        
        [OUTPUT FORMAT]
        Return a JSON Object with a list called 'candidates'. 
        Format:
        {{
          "candidates": [
            {{
              "reasoning": "Brief explanation of why this mix works.",
              "reaction_ids": [0, 5, 12],
              "parameter_values": [[1.0], [0.5], [2.0]] 
            }}
          ]
        }}
        """
        
        config = GenerationConfig(temperature=0.7, response_mime_type="application/json")
        try:
            response = self.smart_model.generate_content(final_prompt, generation_config=config)
            data = json.loads(response.text)
            candidates = data.get("candidates", [])
            
            # RETURN THE TRANSCRIPT TOO
            return candidates, transcript 
            
        except Exception as e:
            print(f"[VertexMultiAgentDebate] JSON Generation Error: {e}")
            # Return empty list and whatever transcript we have so far
            return [], transcript

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
        print(f"[Debate] Simulating candidates...")

        for cand in candidates:
            feedback_entry = {"reasoning": cand.get("reasoning", "No reasoning"), "reward": "N/A", "valid": False}

            try:
                r_ids = cand.get('reaction_ids', [])
                p_vals = cand.get('parameter_values', [])
                
                # --- SAFETY CHECK 1: Length ---
                if len(r_ids) != max_added_reactions:
                    print(f"  -> [Skip] Length Mismatch: Got {len(r_ids)}, Expected {max_added_reactions}")
                    feedback_entry["reasoning"] += f" (Length Mismatch)"
                    self._update_feedback(feedback_entry)
                    continue
                
                if not r_ids or len(r_ids) != len(p_vals):
                    continue

                if is_ordered_policy:
                    paired = sorted(zip(r_ids, p_vals), key=lambda x: x[0])
                else:
                    paired = list(zip(r_ids, p_vals))
                
                sorted_ids, sorted_params = zip(*paired) if paired else ([], [])
                
                gemini_env = Environment(crn_template, max_added_reactions, logger=logger)
                gemini_env.reset()
                
                trajectory_valid = True
                for r_idx, params in zip(sorted_ids, sorted_params):
                    if r_idx >= len(library): 
                        trajectory_valid = False
                        break
                    
                    # --- SAFETY CHECK 2: Sanitize Parameters (Fix for LogNormal Crash) ---
                    # Ensure no parameter is exactly 0.0 or negative. 
                    # RL agent uses LogNormal, which requires x > 0.
                    raw_params_list = list(params) if isinstance(params, (list, tuple)) else [params]
                    sanitized_params = []
                    for p in raw_params_list:
                        val = float(p)
                        if val <= 1e-6:
                            val = 1e-6  # Clamp to small positive value
                        sanitized_params.append(val)
                    
                    # Update the list to use the safe values
                    current_params_list = sanitized_params
                    
                    raw_action_dict = {
                        'reaction index': int(r_idx), 
                        'parameters': current_params_list, 
                        'continuous parameters': current_params_list, 
                        'discrete parameters': None 
                    }
                    
                    action_object = actuator.actuate(raw_action_dict)
                    gemini_env.step(action=action_object, stepper=stepper, raw_action=raw_action_dict)

                if not trajectory_valid: continue

                if len(gemini_env.raw_actions_taken) != max_added_reactions:
                     print("  -> [Skip] Trajectory length mismatch after Sim")
                     continue

                reward_result = compute_reward_func(gemini_env.state)
                
                if isinstance(reward_result, (tuple, list)): raw_val = reward_result[0]
                else: raw_val = reward_result
                if torch.is_tensor(raw_val): r_float = raw_val.item()
                else: r_float = float(raw_val)

                if not hasattr(gemini_env.state, 'last_task_info'): gemini_env.state.last_task_info = {}
                gemini_env.state.last_task_info['reward'] = r_float
                
                valid_envs.append(gemini_env)
                self._update_internal_hof(r_float, gemini_env, cand)
                
                feedback_entry["reward"] = f"{r_float:.6f}"
                feedback_entry["valid"] = True
                feedback_entry["CRN"] = str(gemini_env.state)
    
                self._update_feedback(feedback_entry)
                print(f"  -> Valid. Loss: {r_float:.6f}")

            except Exception as e:
                print(f"  -> Candidate Failed: {e}")
                self._update_feedback(feedback_entry)
        
        return valid_envs

    def _update_feedback(self, entry: Dict):
        """Adds result to short-term memory buffer (FIFO)."""
        self.feedback_buffer.append(entry)
        if len(self.feedback_buffer) > self.feedback_history_size:
            self.feedback_buffer.pop(0)

    def _update_internal_hof(self, reward: float, env, candidate_dict: Dict):
        """Updates the internal 'Best of Gemini' Hall of Fame."""
        entry = {
            "reward": reward,
            "crn_string": str(env.state),
            "original_candidate": candidate_dict
        }
        self.gemini_top_k.append(entry)
        # Sort by Reward (Ascending for Loss)
        self.gemini_top_k.sort(key=lambda x: x['reward'], reverse=False) 
        
        if len(self.gemini_top_k) > self.track_top_k_limit:
            self.gemini_top_k = self.gemini_top_k[:self.track_top_k_limit]

    def get_analysis_data(self):
        """Returns the internal HoF for inspection."""
        return self.gemini_top_k