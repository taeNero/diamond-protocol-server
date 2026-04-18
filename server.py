import os
import traceback
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, Process, LLM
from supabase import create_client, Client
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# 1. Setup & Keys
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")  # <-- Explicitly set
os.environ["CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"] = "true"

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase = None
if url and key:
    supabase = create_client(url, key)

app = Flask(__name__)
# Enable CORS for the entire application to allow cross-origin requests from Lovable
CORS(app) 

# 2. Define the Tools
class SupabaseLogInput(BaseModel):
    message: str = Field(..., description="A short summary of what you just did or found.")
    action_type: str = Field(..., description="Category of action: e.g., 'READ_VAULT', 'SYSTEM_BOOT', 'CLIENT_INTAKE'")
    status: str = Field(default="INFO", description="Status: 'INFO', 'OK', or 'ERROR'")

class SupabaseLoggerTool(BaseTool):
    name: str = "Dashboard Broadcast Tool"
    description: str = "CRITICAL: You must use this tool to log your final answers, actions, and findings so the human can see them on the Command Dashboard."
    args_schema: type[BaseModel] = SupabaseLogInput

    def _run(self, message: str, action_type: str, status: str = "INFO") -> str:
        try:
            if supabase:
                supabase.table("agent_logs").insert({
                    "agent_name": "Diamond",
                    "action_type": action_type,
                    "message": message,
                    "status": status
                }).execute()
            return "Successfully broadcasted to the Command Dashboard."
        except Exception as e:
            return f"Failed to log to database: {str(e)}"

db_logger_tool = SupabaseLoggerTool()

# --- UPGRADED CLIENT PIPELINE TOOL ---
class ClientIntakeInput(BaseModel):
    client_name: str = Field(..., description="Name of the client.")
    package_tier: str = Field(default="LEAD", description="Package bought. Default to 'LEAD' if none.")
    revenue_value: float = Field(default=0.0, description="Amount paid. Default to 0 if none.")

class ClientIntakeTool(BaseTool):
    name: str = "Client Pipeline Tool"
    description: str = "CRITICAL: Use this tool to officially add a new client to the DPC CRM pipeline database when they make a purchase or submit an intake."
    args_schema: type[BaseModel] = ClientIntakeInput

    def _run(self, client_name: str, package_tier: str = "LEAD", revenue_value: float = 0.0) -> str:
        try:
            # Safety checks to handle potential 'None' values passed by the agent
            safe_package = (package_tier or "LEAD").upper()
            safe_revenue = revenue_value if revenue_value is not None else 0.0
            
            if supabase:
                supabase.table("dpc_clients").insert({
                    "client_name": client_name,
                    "package_tier": safe_package,
                    "status": "INTAKE",
                    "progress_percentage": 10, # Starts at 10% for Intake
                    "revenue_value": safe_revenue
                }).execute()
            return f"Successfully added {client_name} to the CRM."
        except Exception as e:
            return f"Failed to add client to CRM: {str(e)}"

pipeline_tool = ClientIntakeTool()

class VaultSearchInput(BaseModel):
    query: str = Field(..., description="The concept, rule, or SOP you need to look up in the DPC vault.")

class VaultSearchTool(BaseTool):
    name: str = "Knowledge Vault Search Tool"
    description: str = "CRITICAL: Search the DPC Obsidian vault for SOPs, frameworks, and rules. Use this whenever you need to know how to handle a specific client, package, or internal process."
    args_schema: type[BaseModel] = VaultSearchInput

    def _run(self, query: str) -> str:
        try:
            # 1. Get embeddings from OpenAI (cloud-based, works on Railway)
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding

            # 2. Search Supabase using the pgvector function
            if supabase:
                result = supabase.rpc(
                    'match_dpc_knowledge', 
                    {'query_embedding': query_embedding, 'match_threshold': 0.5, 'match_count': 2}
                ).execute()

                if not result.data:
                    return "No relevant SOPs found in the vault for that query."

                # 3. Format the results for the agent to read
                knowledge = "--- VAULT KNOWLEDGE FOUND ---\n\n"
                for match in result.data:
                    knowledge += f"File: {match['file_name']}\nContent:\n{match['content']}\n\n"
                
                return knowledge
            return "Database connection failed."
        except Exception as e:
            return f"Error searching the vault: {str(e)}"

#vault_search_tool = VaultSearchTool()


# 3. Define the LLM
claude_llm = LLM(
    model="anthropic/claude-opus-4-1",  # or "anthropic/claude-3-5-sonnet-20241022"
    temperature=0.7,
    max_tokens=4096
)

# 4. Define the Agents 
angel_orchestrator = Agent(
    role='Chief Orchestrator',
    goal='Analyze incoming webhooks and determine which agent should handle them.',
    backstory='You are Angel, the master router for the DPC ecosystem. You analyze payloads and give clear directives.',
    verbose=True,
   # tools=[vault_search_tool], 
    llm=claude_llm
)

diamond_agent = Agent(
    role='Operations Manager',
    goal='Execute internal operations and log updates to the dashboard.',
    backstory='You are the Diamond Protocol agent. You follow Angel\'s orders and use the broadcast tool to report success.',
    verbose=True,
    tools=[db_logger_tool, pipeline_tool],
    llm=claude_llm
)

# 5. The Webhook Listener
@app.route('/intake-webhook', methods=['POST'])
def handle_intake():
    try:
        payload = request.json
        print(f"\n📡 [WEBHOOK RECEIVED]: {payload}\n")
        
        analyze_task = Task(
            description=f"Analyze this incoming webhook payload: {payload}. Identify the client name. If 'package' or 'amount' are null or missing, this is a free Lead. Set the package to 'LEAD' and amount to 0. Pass a summary to the next agent.",
            expected_output="A short summary of the client and their purchase or lead status.",
            agent=angel_orchestrator
        )

        execute_task = Task(
           description="Take Angel's summary. First, use the Client Pipeline Tool to add the client to the database. Ensure you handle missing packages/amounts correctly. Then, use the Dashboard Broadcast Tool to log 'New Intake: [Client Name] starting [Package]' with action_type 'CLIENT_INTAKE' and status 'OK'.",
            expected_output="Confirmation that the client was added to the CRM and the broadcast was logged.",
            agent=diamond_agent
        )

        crew = Crew(
            agents=[angel_orchestrator, diamond_agent],
            tasks=[analyze_task, execute_task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return jsonify({"status": "success", "message": "DPC Swarm executed successfully"}), 200

    except Exception as e:
        print("\n❌ SERVER CRASHED DURING WEBHOOK:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/cathedral-webhook', methods=['POST'])
def handle_cathedral():
    try:
        data = request.json
        print(f"\n🏛️ [CATHEDRAL EVENT]: {data}\n", flush=True)
        
        # Store engagement data in Supabase
        supabase.table("cathedral_engagement").insert({
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "rank_level": data.get("rank_level"),
            "value": data.get("value"),
            "event_type": data.get("event", "rank_up"),
            "test_mode": data.get("test", False)
        }).execute()
        
        # Log to agent_logs for dashboard feed
        supabase.table("agent_logs").insert({
            "agent_name": "Cathedral",
            "action_type": "RANK_UP",
            "message": f"{data.get('username')} ranked up to {data.get('rank_level')} (Value: {data.get('value')})",
            "status": "OK"
        }).execute()
        
        return jsonify({"status": "success", "message": "Cathedral engagement tracked"}), 200
    except Exception as e:
        print(f"\n❌ CATHEDRAL WEBHOOK ERROR: {str(e)}\n", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/engagement', methods=['GET'])
def get_engagement_data():
    try:
        # Get recent engagement data from Supabase
        result = supabase.table("cathedral_engagement").select("*").order("created_at", desc=True).limit(100).execute()
        
        # Aggregate by rank level (for cohorts)
        rank_stats = {}
        for row in result.data:
            rank = row.get('rank_level', 'unknown')
            if rank not in rank_stats:
                rank_stats[rank] = {'count': 0, 'total_value': 0}
            rank_stats[rank]['count'] += 1
            rank_stats[rank]['total_value'] += float(row.get('value') or 0)
        
        # Format as cohorts (matches UI expectations)
        cohorts = []
        for rank_name, stats in rank_stats.items():
            cohorts.append({
                "name": rank_name.title(),  # "seeker" → "Seeker"
                "completion": min(100, int(stats['total_value'] / 10)),  # Calculate % based on value
                "users": stats['count']
            })
        
        # Sort by completion (highest first)
        cohorts.sort(key=lambda x: x['completion'], reverse=True)
        
        # Generate recent coupons/rewards (from agent_logs or separate table)
        coupons_result = supabase.table("agent_logs").select("message, created_at").eq("action_type", "RANK_UP").order("created_at", desc=True).limit(5).execute()
        
        coupons = []
        for row in coupons_result.data:
            coupons.append({
                "code": f"RANK-{row['created_at'][:10].replace('-', '')}",  # Generate pseudo-code
                "reward": row['message'][:50],  # Truncate message
                "triggered": row['created_at']
            })
        
        return jsonify({
            "cohorts": cohorts,
            "coupons": coupons
        }), 200
        
    except Exception as e:
        print(f"\n❌ ENGAGEMENT API ERROR: {str(e)}\n", flush=True)
        return jsonify({
            "cohorts": [],
            "coupons": []
        }), 200  # Return 200 with empty data (graceful degradation)


if __name__ == '__main__':
    print("🎧 Angel Orchestrator is online and listening on port 5000...")
app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
