import time
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from typing import List, Dict, Any, Optional

class NLPAgent:
    def __init__(self, 
                 name: str, 
                 role_prompt: str, 
                 model_name: str = "gemini-1.5-pro-preview-0409", 
                 temperature: float = 0.5,
                 response_mime_type: str = "text/plain"):
        
        self.name = name
        self.role_prompt = role_prompt 
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Vertex AI Model
        self.model = GenerativeModel(model_name)
        
        # Store config safely
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=65536,
            response_mime_type=response_mime_type
        )
        
        # Graph connections
        self.listening_to: List['NLPAgent'] = []
        self.memory: List[Dict] = []
        self.latest_output: str = ""

    def listen_to(self, agents: List['NLPAgent']):
        """Registers which agents this agent listens to."""
        self.listening_to.extend(agents)

    def think(self, context_prompt: str) -> str:
        """
        Generates a response based on context and inputs from upstream agents.
        Includes retry logic for network stability.
        """
        # 1. Gather what I've heard
        script = ""
        for agent in self.listening_to:
            if agent.latest_output:
                script += f"**{agent.name} ({agent.role_prompt[:20]}...)**: {agent.latest_output}\n\n"

        if not script:
            script = "(No one has spoken yet.)"

        # 2. Build Prompt
        full_prompt = f"""
        [SYSTEM ROLE]
        {self.role_prompt}

        [ENVIRONMENT CONTEXT]
        {context_prompt}

        [CURRENT CONVERSATION]
        {script}

        [INSTRUCTION]
        Digest the context and the conversation history. 
        Output your response clearly. 
        """

        # 3. Generate with Retry Logic
        content = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    full_prompt, 
                    generation_config=self.generation_config
                )
                content = response.text.strip()
                break # Success
                
            except Exception as e:
                print(f"[{self.name}] Connection Error (Attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    # Final Fallback
                    print(f"[{self.name}] Failed after retries. Returning empty.")
                    content = "{}" 

        # 4. Update Memory
        entry = {
            "timestamp": time.time(),
            "script_seen": script,
            "content": content
        }
        self.memory.append(entry)
        self.latest_output = content
        
        return content


class DebateGraph:
    """
    Manages the topology and execution order of NLPAgents.
    """
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.agents: Dict[str, NLPAgent] = {}
        self.execution_order: List[str] = []
        
        # Init Vertex AI globally
        vertexai.init(project=project_id, location=location)

    def add_agent(self, agent: NLPAgent):
        self.agents[agent.name] = agent

    def set_execution_order(self, agent_names: List[str]):
        self.execution_order = agent_names

    def set_output_nodes(self, agent_names: List[str]):
        self.output_nodes = agent_names

    def run_epoch(self, default_context: str, specific_contexts: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Runs one full pass of the debate graph.
        """
        if specific_contexts is None:
            specific_contexts = {}

        transcript = ""
        outputs = {}

        print(f"--- [DebateGraph] Starting Epoch ---")

        for name in self.execution_order:
            if name in self.agents:
                agent = self.agents[name]
                print(f" -> {name} is thinking...")
                
                # Determine context
                current_ctx = default_context
                if name in specific_contexts:
                    print(f"    (Using specialized context for {name})")
                    current_ctx = specific_contexts[name]

                # Execute
                response = agent.think(current_ctx)
                
                # Log
                transcript += f"\n\n=== {name} ===\n{response}"
                outputs[name] = response
            else:
                print(f"Warning: Agent {name} in execution order but not found in graph.")

        return {
            "transcript": transcript,
            "outputs": outputs
        }