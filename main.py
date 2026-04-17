import os
os.environ["CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"] = "true"
from crewai import Agent, Task, Crew, Process, LLM
from supabase import create_client, Client
from crewai_tools import FileReadTool
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# 1. Load keys from .env securely
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


# 2. Connect to Supabase
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
if url and key: 
    supabase: Client = create_client(url, key)
print("✅ Connected to Supabase Vault")

# Define the expected inputs for the tool
class SupabaseLogInput(BaseModel):
    message: str = Field(..., description="A short summary of what you just did or found.")
    action_type: str = Field(..., description="Category of action: e.g., 'READ_VAULT', 'SYSTEM_BOOT'")
    status: str = Field(default="INFO", description="Status: 'INFO', 'OK', or 'ERROR'")

# Build the custom broadcast tool
class SupabaseLoggerTool(BaseTool):
    name: str = "Dashboard Broadcast Tool"
    description: str = "CRITICAL: You must use this tool to log your final answers, actions, and findings so the human can see them on the Command Dashboard."
    args_schema: type[BaseModel] = SupabaseLogInput

    def _run(self, message: str, action_type: str, status: str = "INFO") -> str:
        try:
            # Pushes the log to the Supabase table
            supabase.table("agent_logs").insert({
                "agent_name": "Diamond",
                "action_type": action_type,
                "message": message,
                "status": status
            }).execute()
            return "Successfully broadcasted to the Command Dashboard."
        except Exception as e:
            return f"Failed to log to database: {str(e)}"

# Instantiate the tool
db_logger_tool = SupabaseLoggerTool()
# 3. Create the LLM with CURRENT available model (Claude 4.x)
claude_llm = LLM(
    model="anthropic/claude-sonnet-4-6",  # ✅ Current available model
    temperature=0.7,
    max_tokens=4096
)

# 4. Create the Obsidian Tool
sop_path = r"C:\Users\diner\.openclaw\workspace\Dalpazor-Vault\Operations\Diamond_Protocol_SOP.md"
sop_reader_tool = FileReadTool(file_path=sop_path)

# 5. Define the Agent
diamond_agent = Agent(
    role='Operations Manager',
    goal='Manage the DPC client pipeline and internal operations efficiently.',
    backstory='You are the Diamond Protocol agent, responsible for the internal health of DPC. You strictly follow the SOPs in the Obsidian vault.',
    verbose=True,
    allow_delegation=False,
    llm=claude_llm,
    tools=[sop_reader_tool, db_logger_tool] # <-- FIX 1: Handed the agent the DB tool
)

# 6. Define the Task (You can delete the old audit_task entirely)
ignition_task = Task(
    description="Read your SOP document using the sop_reader_tool. Summarize your exact capabilities and constraints. Then, you MUST use the Dashboard Broadcast Tool to log a message saying 'SOP successfully read and initialized' with the action_type 'SYSTEM_BOOT' and status 'OK'.",
    expected_output="A bulleted summary of the agent's capabilities, followed by a confirmation that the log was broadcasted.",
    agent=diamond_agent
)

# 7. Ignite the Agent
crew = Crew(
    agents=[diamond_agent],
    tasks=[ignition_task], # <-- FIX 2: Pointed the crew to the new task
    process=Process.sequential
)

if __name__ == "__main__":
    print("🚀 Booting Diamond Agent...")
    result = crew.kickoff()
    print("\n\n### AGENT REPORT ###\n")
    print(result)