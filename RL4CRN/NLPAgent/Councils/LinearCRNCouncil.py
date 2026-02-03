from typing import List, Dict, Any, Tuple, Optional
from RL4CRN.NLPAgent.CRNDebate import CRNDebate
from RL4CRN.NLPAgent.nlp_core import NLPAgent
import json

class LinearCRNCouncil(CRNDebate):
    """
    Implements a linear/waterfall debate structure:
    Narrator -> Opportunist -> Contrarian -> Skeptic -> Player
    """
    def __init__(self, project_id: str, location: str = "global"):
        super().__init__(project_id, location)
        self.setup_agents()

    def setup_agents(self):
        # 1. Instantiate Agents
        self.narrator = NLPAgent("Narrator", 
            role_prompt="You are the NARRATOR. Summarize the Hall of Fame objectively. Do NOT suggest solutions."
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what." 
                        " This means you need to pick N additional reactions from the library to add to the template."
                        " You cannot use any input (like u_1) in the reaction rates you select."
                        " The scores you will see later are LOSS values: lower is better! And remember to describe again the task! Look also for the literature for known motifs.",
            model_name="gemini-3-pro-preview", temperature=0.3)
        
        self.opportunist = NLPAgent("Opportunist", 
            role_prompt="You are the OPPORTUNIST. Propose five conservative modifications (full CRNs). 'Copy smart, don't reinvent.'"
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what. " 
                        " This means you need to pick N additional reactions from the library to add to the template."
                        " You cannot use any input (like u_1) in the reaction rates you select.",
            model_name="gemini-2.5-flash", temperature=0.5)
        
        self.contrarian = NLPAgent("Contrarian", 
            role_prompt="You are the CONTRARIAN. Propose five radical alternatives ignoring safety (full CRNs). However, be still effective, remember your goal!"
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what. This means you need to pick N additional reactions" 
                        " from the library to add to the template."
                        " You cannot use any input (like u_1) in the reaction rates you select.",
            model_name="gemini-2.5-flash", temperature=0.9)
        
        self.skeptic = NLPAgent("Skeptic", 
            role_prompt="You are the SKEPTIC. Critique the others based on Control Theory and Physics. Suggest solution based on the known literature."
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what. " 
                        " This means you need to pick N additional reactions from the library to add to the template."
                        " You cannot use any input (like u_1) in the reaction rates you select.",
            model_name="gemini-2.5-flash", temperature=0.2)
        
        self.player = NLPAgent("Player", 
            role_prompt="You are the PLAYER. Listen to all the others and synthesize their inputs. Select the best ideas and compile them into ten candidate CRNs. " 
                        " You can also provide reasoning for your choices and describe how to pick parameters correctly." \
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what. " 
                        " This means you need to pick N additional reactions from the library to add to the template." 
                        " You cannot use any input (like u_1) in the reaction rates you select."
                        " You need to specify how to select the rates (the parameter values) for each reaction you choose, and say why. Be precise and use Control Theory concepts."
                        " You must not select again reactions already present in the template as they will be there no matter what.",
            model_name="gemini-3-pro-preview", temperature=0.7, response_mime_type="application/json")
        
        self.writer = NLPAgent("Writer", 
            role_prompt="You are the WRITER. Compile the final candidate list into valid JSON format."
                        " Remember that the reactions in the template do not count towards the max reactions allowed and they will be there no matter what. " 
                        " This means you need to pick N additional reactions from the library to add to the template."
                        " You cannot use any input (like u_1) in the reaction rates you select."
                        " You must not select again reactions already present in the template."
                        " Propose many chemical reaction networks (CRNs) as candidates, each with a set of reaction IDs and parameter values."
                        # " For fairness, your highest rate must not exceed 10. If you suggest a rate higher than 10, normalize all rates so that the highest is exactly 10."
            ,
            model_name="gemini-2.5-flash", temperature=0.7, response_mime_type="application/json")

        # 2. Register Nodes
        for agent in [self.narrator, self.opportunist, self.contrarian, self.skeptic, self.player, self.writer]:
            self.add_agent(agent)

        # 3. Define Topology (Edges)
        self.narrator.listen_to([]) 
        self.opportunist.listen_to([self.narrator])
        self.contrarian.listen_to([self.narrator, self.opportunist])
        self.skeptic.listen_to([self.narrator, self.opportunist, self.contrarian])
        self.player.listen_to([self.narrator, self.opportunist, self.contrarian, self.skeptic])
        self.writer.listen_to([self.player])

        # 4. Define Execution Flow
        self.set_execution_order(["Narrator", "Opportunist", "Contrarian", "Skeptic", "Player", "Writer"])
        self.set_output_nodes(["Writer"])

    # In RL4CRN/NLPAgent/Councils/LinearCRNCouncil.py

    def run_debate_session(self, 
                           task_desc: str, 
                           hof_iter, 
                           library_description: str,
                           library_explicit_str: str,
                           max_added_reactions: int) -> Tuple[List[Dict], str]:
        
        hof_str = self._format_hof(hof_iter)
        feedback_str = self._format_feedback()
        
        # --- Context A: The "Strategy" Board (Lightweight) ---
        # Used by: Narrator, Opportunist, Contrarian, Skeptic, Player
        reasoning_context = f"""
        [TASK MISSION]
        {task_desc}
        
        [CONSTRAINT]
        You must use EXACTLY {max_added_reactions} reactions.

        [HALL OF FAME (Past Performance)]
        {hof_str}

        [RECENT FEEDBACK]
        {feedback_str}

        [AVAILABLE CHEMISTRY (Concept)]
        {library_description}
        *Do not worry about specific reaction indices yet. Discuss the topology and species dynamics.*
        """

        # --- Context B: The "Syntax" Board (Heavy) ---
        # Used by: Writer ONLY
        writer_context = f"""
        [TASK MISSION]
        {task_desc} (Format Output Only)
        
        [CONSTRAINT]
        Map the Player's requested reactions to the indices in the Library below.

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

        [REACTION LIBRARY (Explicit)]
        {library_explicit_str}
        """

        # --- Run the Graph with Overrides ---
        # Everyone gets reasoning_context, EXCEPT the Writer who gets writer_context
        context_map = {
            "Writer": writer_context
        }

        result = self.run_epoch(default_context=reasoning_context, specific_contexts=context_map)
        
        # --- Parse Output ---
        writer_json = result['outputs'].get('Writer', '{}')
        candidates = []
        try:
            # Clean up potential markdown formatting from LLM
            clean_json = writer_json.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            # Now we can safely access "candidates"
            candidates = data.get("candidates", [])
            
        except json.JSONDecodeError:
            print(f"[LinearCRNCouncil] JSON Decode Error. Received: {writer_json[:50]}...")
        except Exception as e:
            print(f"[LinearCRNCouncil] Unexpected Error: {e}")

        return candidates, result['transcript']