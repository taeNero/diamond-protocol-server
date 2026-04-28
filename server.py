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
from datetime import datetime, timezone

# ============================================================
# 1. SETUP & KEYS
# ============================================================
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"] = "true"

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase = None
if url and key:
    supabase = create_client(url, key)

app = Flask(__name__)
CORS(app)


# ============================================================
# 2. TOOLS
# ============================================================

# --- Dashboard Broadcast Tool (unchanged) ---
class SupabaseLogInput(BaseModel):
    message: str = Field(..., description="A short summary of what you just did or found.")
    action_type: str = Field(..., description="Category of action: e.g., 'READ_VAULT', 'SYSTEM_BOOT', 'CLIENT_INTAKE'")
    status: str = Field(default="INFO", description="Status: 'INFO', 'OK', or 'ERROR'")

class SupabaseLoggerTool(BaseTool):
    name: str = "Dashboard Broadcast Tool"
    description: str = "CRITICAL: Use this tool to log your final answers, actions, and findings so the human can see them on the Command Dashboard."
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


# --- Client Pipeline Tool (corrected) ---
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
            safe_package = (package_tier or "LEAD").upper()
            safe_revenue = revenue_value if revenue_value is not None else 0.0
            if supabase:
                supabase.table("dpc_clients").insert({
                    "client_name": client_name,
                    "package_tier": safe_package,
                    "status": "INTAKE",
                    "progress_percentage": 10,
                    "revenue_value": safe_revenue
                }).execute()
            return f"Successfully added {client_name} to the CRM."
        except Exception as e:
            return f"Failed to add client to CRM: {str(e)}"

pipeline_tool = ClientIntakeTool()


# --- Supabase Data Reader Tool (NEW) ---
# Allows Guardian agents to read raw data from any table for analysis.
class SupabaseReadInput(BaseModel):
    table: str = Field(..., description="The Supabase table to read from.")
    limit: int = Field(default=50, description="Number of rows to retrieve.")
    order_by: str = Field(default="created_at", description="Column to order by.")

class SupabaseReaderTool(BaseTool):
    name: str = "Supabase Data Reader"
    description: str = "Read raw data from a Supabase table for analysis. Use this to fetch cathedral_engagement, shopify_orders, dpc_clients, or any other table."
    args_schema: type[BaseModel] = SupabaseReadInput

    def _run(self, table: str, limit: int = 50, order_by: str = "created_at") -> str:
        try:
            if supabase:
                result = supabase.table(table).select("*").order(order_by, desc=True).limit(limit).execute()
                if not result.data:
                    return f"No data found in table '{table}'."
                return str(result.data)
            return "Database connection unavailable."
        except Exception as e:
            return f"Error reading from {table}: {str(e)}"

supabase_reader_tool = SupabaseReaderTool()


# --- Guardian Metrics Writer Tool (NEW) ---
# Writes processed Guardian intelligence back to Supabase.
# Supabase table needed: guardian_metrics
# Schema: id (uuid), created_at (timestamptz), guardian (text),
#         metric_key (text), metric_value (float), summary (text), session_date (date)
class GuardianMetricsInput(BaseModel):
    guardian: str = Field(..., description="Guardian name: 'kael', 'siryandorin', 'anansi', or 'aurixen'")
    metric_key: str = Field(..., description="The metric being recorded, e.g. 'engagement_score', 'revenue_leverage', 'narrative_density', 'risk_index'")
    metric_value: float = Field(..., description="Numeric value of the metric.")
    summary: str = Field(..., description="A brief natural language summary of the Guardian's analysis and what the metric means.")

class GuardianMetricsWriterTool(BaseTool):
    name: str = "Guardian Metrics Writer"
    description: str = "CRITICAL: After completing your analysis, use this tool to write your processed metrics and insights into the guardian_metrics table in Supabase so the Council can read them."
    args_schema: type[BaseModel] = GuardianMetricsInput

    def _run(self, guardian: str, metric_key: str, metric_value: float, summary: str) -> str:
        try:
            if supabase:
                supabase.table("guardian_metrics").insert({
                    "guardian": guardian.lower(),
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "summary": summary,
                    "session_date": datetime.now(timezone.utc).date().isoformat()
                }).execute()

                # Also broadcast to dashboard feed
                supabase.table("agent_logs").insert({
                    "agent_name": guardian.upper(),
                    "action_type": "GUARDIAN_SYNC",
                    "message": f"{guardian.title()} updated {metric_key}: {metric_value} — {summary[:80]}",
                    "status": "OK"
                }).execute()

            return f"Successfully wrote {guardian} metrics to Supabase."
        except Exception as e:
            return f"Failed to write Guardian metrics: {str(e)}"

guardian_metrics_tool = GuardianMetricsWriterTool()


# ============================================================
# 3. LLM
# ============================================================
claude_llm = LLM(
    model="anthropic/claude-opus-4-1",
    temperature=0.7,
    max_tokens=4096
)


# ============================================================
# 4. AGENTS
# ============================================================

# --- Existing Agents (unchanged) ---
angel_orchestrator = Agent(
    role='Chief Orchestrator',
    goal='Analyze incoming webhooks and determine which agent should handle them.',
    backstory='You are Angel, the master router for the DPC ecosystem. You analyze payloads and give clear directives.',
    verbose=True,
    llm=claude_llm
)

diamond_agent = Agent(
    role='Operations Manager',
    goal='Execute internal operations and log updates to the dashboard.',
    backstory="You are the Diamond Protocol agent. You follow Angel's orders and use the broadcast tool to report success.",
    verbose=True,
    tools=[db_logger_tool, pipeline_tool],
    llm=claude_llm
)

# --- Guardian Agents (NEW) ---

kael_agent = Agent(
    role='Kael — Guardian of Performance & Energy',
    goal='Analyze Cathedral engagement data to calculate the collective energy and performance score of the DPC community. Identify consistency, momentum, and high-value interactions.',
    backstory=(
        "You are Kael, the Guardian of Performance and Energy within the Diamond Protocol. "
        "Your domain is the somatic and operational vitality of the DPC ecosystem. "
        "You read raw Cathedral engagement data — rank events, activity values, user consistency — "
        "and distill it into a single Energy Score that tells the Architect how alive and active the community is. "
        "High engagement with consistent rank progressions = high energy. Gaps and stalls = low energy. "
        "You write your findings to the guardian_metrics table so the Council can synthesize them."
    ),
    verbose=True,
    tools=[supabase_reader_tool, guardian_metrics_tool, db_logger_tool],
    llm=claude_llm
)

siryandorin_agent = Agent(
    role='Siryandorin — Guardian of Revenue & Leverage',
    goal='Analyze Shopify orders and client pipeline data to calculate revenue leverage, conversion rates, and economic momentum for the DPC ecosystem.',
    backstory=(
        "You are Siryandorin, the Guardian of Revenue and Leverage within the Diamond Protocol. "
        "Your domain is the economic resonance of DPC. You read Shopify order data and the client pipeline "
        "to calculate total revenue, average order value, lead-to-client conversion rate, and revenue leverage ratio. "
        "You identify which package tiers are performing, where revenue is accelerating or stalling, "
        "and what the economic trajectory looks like. You write your metrics to the guardian_metrics table."
    ),
    verbose=True,
    tools=[supabase_reader_tool, guardian_metrics_tool, db_logger_tool],
    llm=claude_llm
)

anansi_agent = Agent(
    role='Anansi — Guardian of Brand & Community',
    goal='Analyze the narrative density and community health of DPC by examining lead sources, intake velocity, and engagement patterns across all properties.',
    backstory=(
        "You are Anansi, the Guardian of Brand and Community within the Diamond Protocol. "
        "Your domain is the story and resonance of DPC in the world. "
        "You analyze lead intake velocity, the ratio of leads to paying clients, which packages attract the most interest, "
        "and the overall narrative health of the pipeline. "
        "A strong brand shows high intake velocity, diverse lead sources, and growing community engagement. "
        "You calculate a Narrative Density score and write your findings to the guardian_metrics table."
    ),
    verbose=True,
    tools=[supabase_reader_tool, guardian_metrics_tool, db_logger_tool],
    llm=claude_llm
)

aurixen_agent = Agent(
    role='Aurixen — Guardian of Strategy & Risk',
    goal='Synthesize data across all domains to identify strategic risks, environmental shifts, and highest-leverage opportunities for the DPC ecosystem.',
    backstory=(
        "You are Aurixen, the Guardian of Strategy and Risk within the Diamond Protocol. "
        "Your domain is futures, threats, and opportunities. You read the outputs from all other Guardians "
        "and cross-reference them with the raw data to identify conflicts, blind spots, and emerging risks. "
        "Low energy + high revenue = burnout risk. High leads + low conversion = funnel risk. "
        "High engagement + low revenue = monetization gap. You calculate a Risk Index and Strategic Opportunity score "
        "and write your synthesis to the guardian_metrics table."
    ),
    verbose=True,
    tools=[supabase_reader_tool, guardian_metrics_tool, db_logger_tool],
    llm=claude_llm
)


# ============================================================
# 5. ROUTES — EXISTING (unchanged)
# ============================================================

@app.route('/intake-webhook', methods=['POST'])
def handle_intake():
    try:
        payload = request.json
        print(f"📡 [WEBHOOK RECEIVED]: {payload}")

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
        print("❌ SERVER CRASHED DURING WEBHOOK:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cathedral-webhook', methods=['POST'])
def handle_cathedral():
    try:
        data = request.json
        print(f"🏛️ [CATHEDRAL EVENT]: {data}", flush=True)

        supabase.table("cathedral_engagement").insert({
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "rank_level": data.get("rank_level"),
            "value": data.get("value"),
            "event_type": data.get("event", "rank_up"),
            "test_mode": data.get("test", False)
        }).execute()

        supabase.table("agent_logs").insert({
            "agent_name": "Cathedral",
            "action_type": "RANK_UP",
            "message": f"{data.get('username')} ranked up to {data.get('rank_level')} (Value: {data.get('value')})",
            "status": "OK"
        }).execute()

        return jsonify({"status": "success", "message": "Cathedral engagement tracked"}), 200
    except Exception as e:
        print(f"❌ CATHEDRAL WEBHOOK ERROR: {str(e)}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/engagement', methods=['GET'])
def get_engagement_data():
    try:
        result = supabase.table("cathedral_engagement").select("*").order("created_at", desc=True).limit(100).execute()

        rank_stats = {}
        for row in result.data:
            rank = row.get('rank_level', 'unknown')
            if rank not in rank_stats:
                rank_stats[rank] = {'count': 0, 'total_value': 0}
            rank_stats[rank]['count'] += 1
            rank_stats[rank]['total_value'] += float(row.get('value') or 0)

        cohorts = []
        for rank_name, stats in rank_stats.items():
            cohorts.append({
                "name": rank_name.title(),
                "completion": min(100, int(stats['total_value'] / 10)),
                "users": stats['count']
            })

        cohorts.sort(key=lambda x: x['completion'], reverse=True)

        coupons_result = supabase.table("agent_logs").select("message, created_at").eq("action_type", "RANK_UP").order("created_at", desc=True).limit(5).execute()

        coupons = []
        for row in coupons_result.data:
            coupons.append({
                "code": f"RANK-{row['created_at'][:10].replace('-', '')}",
                "reward": row['message'][:50],
                "triggered": row['created_at']
            })

        return jsonify({"cohorts": cohorts, "coupons": coupons}), 200

    except Exception as e:
        print(f"❌ ENGAGEMENT API ERROR: {str(e)}", flush=True)
        return jsonify({"cohorts": [], "coupons": []}), 200


@app.route('/shopify-webhook', methods=['POST'])
def handle_shopify():
    try:
        data = request.json
        print(f"🛒 [SHOPIFY ORDER]: {data}", flush=True)

        supabase.table("shopify_orders").insert({
            "order_id": data.get("id"),
            "customer_name": data.get("customer", {}).get("name"),
            "total_price": data.get("total_price"),
            "created_at": data.get("created_at")
        }).execute()

        supabase.table("agent_logs").insert({
            "agent_name": "Shopify",
            "action_type": "ORDER_RECEIVED",
            "message": f"Order from {data.get('customer', {}).get('name')}: ${data.get('total_price')}",
            "status": "OK"
        }).execute()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# 6. ROUTES — NEW GUARDIAN ENDPOINTS
# ============================================================

@app.route('/guardian-sync', methods=['POST'])
def run_guardian_sync():
    """
    Triggers all four Guardians to read raw data, process their domain metrics,
    and write results to guardian_metrics in Supabase.
    Call this on a schedule (e.g. daily cron) or manually from the dashboard.
    """
    try:
        print("⚡ [GUARDIAN SYNC INITIATED]", flush=True)

        kael_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch the latest 100 rows from 'cathedral_engagement'. "
                "Analyze the data: count total events, calculate average value, identify the most active rank levels, "
                "and assess consistency of engagement over time. "
                "Calculate an Energy Score from 0-100 based on volume and value of engagement. "
                "Then use the Guardian Metrics Writer to save: metric_key='energy_score', your calculated score, "
                "and a 1-2 sentence summary of community energy. "
                "Finally use the Dashboard Broadcast Tool to log your completion with action_type 'GUARDIAN_SYNC'."
            ),
            expected_output="Confirmation that energy_score has been written to guardian_metrics.",
            agent=kael_agent
        )

        siryandorin_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch the latest 100 rows from 'shopify_orders' and 50 rows from 'dpc_clients'. "
                "Calculate: total revenue from orders, number of paying clients (package_tier != 'LEAD'), "
                "lead-to-client conversion rate, and average order value. "
                "Calculate a Revenue Leverage score from 0-100 based on conversion rate and revenue momentum. "
                "Use the Guardian Metrics Writer to save: metric_key='revenue_leverage', your score, and a summary. "
                "Also save metric_key='conversion_rate' with the actual percentage as the value. "
                "Log completion with the Dashboard Broadcast Tool."
            ),
            expected_output="Confirmation that revenue_leverage and conversion_rate have been written to guardian_metrics.",
            agent=siryandorin_agent
        )

        anansi_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch 50 rows from 'dpc_clients' and 50 rows from 'agent_logs'. "
                "Analyze: lead intake velocity (how many new leads recently), diversity of package tiers, "
                "ratio of organic vs intake leads, and overall narrative health of the pipeline. "
                "Calculate a Narrative Density score from 0-100 reflecting brand resonance and community growth momentum. "
                "Use the Guardian Metrics Writer to save: metric_key='narrative_density', your score, and a summary. "
                "Log completion with the Dashboard Broadcast Tool."
            ),
            expected_output="Confirmation that narrative_density has been written to guardian_metrics.",
            agent=anansi_agent
        )

        aurixen_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch the latest rows from 'guardian_metrics' to read what Kael, "
                "Siryandorin, and Anansi have just written. Also read 'dpc_clients' and 'shopify_orders' for raw context. "
                "Cross-reference all domains: identify conflicts (e.g. high energy but low revenue = monetization gap), "
                "risks (e.g. high leads but low conversion = funnel leak), and strategic opportunities. "
                "Calculate a Risk Index from 0-100 (higher = more risk) and a Strategic Opportunity score from 0-100. "
                "Use the Guardian Metrics Writer twice: once for metric_key='risk_index' and once for metric_key='opportunity_score'. "
                "Log completion with the Dashboard Broadcast Tool."
            ),
            expected_output="Confirmation that risk_index and opportunity_score have been written to guardian_metrics.",
            agent=aurixen_agent
        )

        guardian_crew = Crew(
            agents=[kael_agent, siryandorin_agent, anansi_agent, aurixen_agent],
            tasks=[kael_task, siryandorin_task, anansi_task, aurixen_task],
            process=Process.sequential
        )

        result = guardian_crew.kickoff()
        return jsonify({"status": "success", "message": "Guardian sync complete. Metrics written to Supabase."}), 200

    except Exception as e:
        print("❌ GUARDIAN SYNC FAILED:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/guardian-metrics', methods=['GET'])
def get_guardian_metrics():
    """
    Returns the latest metrics for all four Guardians.
    Used by Flowise to inject live context before a Council Session.
    """
    try:
        result = supabase.table("guardian_metrics").select("*").order("created_at", desc=True).limit(50).execute()

        # Organize by guardian
        metrics = {}
        for row in result.data:
            guardian = row.get("guardian")
            if guardian not in metrics:
                metrics[guardian] = []
            metrics[guardian].append({
                "metric_key": row.get("metric_key"),
                "metric_value": row.get("metric_value"),
                "summary": row.get("summary"),
                "session_date": row.get("session_date")
            })

        return jsonify({"status": "success", "guardian_metrics": metrics}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/council-briefing', methods=['GET'])
def get_council_briefing():
    """
    Returns a pre-built council briefing snapshot combining all live data.
    Flowise calls this endpoint at the start of every Council Session
    to inject real context before the Guardians speak.
    """
    try:
        # Latest clients
        clients = supabase.table("dpc_clients").select("*").order("created_at", desc=True).limit(10).execute()
        leads = [c for c in clients.data if c.get("package_tier") == "LEAD"]
        paying = [c for c in clients.data if c.get("package_tier") != "LEAD"]

        # Latest orders
        orders = supabase.table("shopify_orders").select("*").order("created_at", desc=True).limit(10).execute()
        total_revenue = sum(float(o.get("total_price") or 0) for o in orders.data)

        # Latest engagement
        engagement = supabase.table("cathedral_engagement").select("*").order("created_at", desc=True).limit(20).execute()

        # Latest guardian metrics
        gm = supabase.table("guardian_metrics").select("*").order("created_at", desc=True).limit(20).execute()
        guardian_summary = {}
        for row in gm.data:
            g = row.get("guardian")
            if g not in guardian_summary:
                guardian_summary[g] = {}
            guardian_summary[g][row.get("metric_key")] = {
                "value": row.get("metric_value"),
                "summary": row.get("summary")
            }

        briefing = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": {
                "total_leads": len(leads),
                "total_paying_clients": len(paying),
                "recent_intakes": [c.get("client_name") for c in clients.data[:5]]
            },
            "revenue": {
                "recent_orders": len(orders.data),
                "total_recent_revenue": round(total_revenue, 2)
            },
            "engagement": {
                "recent_events": len(engagement.data),
                "latest_rank_event": engagement.data[0] if engagement.data else None
            },
            "guardian_metrics": guardian_summary
        }

        return jsonify({"status": "success", "briefing": briefing}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# 7. BOOT
# ============================================================
if __name__ == '__main__':
    print("🎧 Angel Orchestrator is online and listening on port 5000...")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
