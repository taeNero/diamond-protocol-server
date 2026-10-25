import os
import re
import json
import hmac
import base64
import hashlib
import threading
import traceback
from functools import wraps
from datetime import datetime, timezone, timedelta

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from supabase import create_client
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
# --- Anthropic SDK no longer accepts `temperature`; CrewAI still sends it. Strip it.
try:
    from anthropic.resources.messages import Messages as _Msgs, AsyncMessages as _AsyncMsgs

    _orig_create = _Msgs.create
    _orig_acreate = _AsyncMsgs.create

    def _create_no_temp(self, *args, **kwargs):
        kwargs.pop("temperature", None)
        return _orig_create(self, *args, **kwargs)

    async def _acreate_no_temp(self, *args, **kwargs):
        kwargs.pop("temperature", None)
        return await _orig_acreate(self, *args, **kwargs)

    _Msgs.create = _create_no_temp
    _AsyncMsgs.create = _acreate_no_temp
except Exception as e:
    print(f"⚠️ temperature patch not applied: {e}", flush=True)
# ============================================================
# 1. SETUP & KEYS
# ============================================================
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠️ ANTHROPIC_API_KEY not set — agent routes will fail until it is.", flush=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DPC_WEBHOOK_SECRET = os.getenv("DPC_WEBHOOK_SECRET")
SHOPIFY_WEBHOOK_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"❌ Supabase client failed to initialize: {e}", flush=True)
else:
    print("⚠️ SUPABASE_URL / SUPABASE_KEY not set — database routes will return 503.", flush=True)

app = Flask(__name__)
CORS(app)


# ============================================================
# 2. HELPERS
# ============================================================

def utc_today():
    return datetime.now(timezone.utc).date()


def log_event(agent_name, action_type, message, status="INFO"):
    """Write to agent_logs without ever crashing the caller."""
    if not supabase:
        print(f"[log skipped — no db] {agent_name} {action_type}: {message}", flush=True)
        return
    try:
        supabase.table("agent_logs").insert({
            "agent_name": agent_name,
            "action_type": action_type,
            "message": message,
            "status": status
        }).execute()
    except Exception as e:
        print(f"❌ log_event failed: {e}", flush=True)


def parse_money(value):
    """Turns 3000, '3000', '$3,000', None into a float safely."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def clean_email(raw):
    """Handles 'Name <email@x.com>' and normalizes case."""
    if not raw:
        return None
    match = re.search(r"<([^>]+)>", str(raw))
    email = match.group(1) if match else str(raw)
    email = email.strip().lower()
    return email or None


def run_in_background(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()


def send_email(recipient, subject, body):
    """Deterministic send via Resend. Returns (ok, info)."""
    api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("DPC_EMAIL_USER", "frontdesk@daplayerscollective.com")

    if not api_key:
        return False, "RESEND_API_KEY missing"

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "from": f"Anansi | Da Players Collective <{sender}>",
                "to": [recipient],
                "subject": subject,
                "text": body
            },
            timeout=15
        )
        if 200 <= response.status_code < 300:
            return True, response.json().get("id", "unknown")
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# --- Business event routing (Oct 2026) ---------------------------------
# Payments/bookings/invoices go to dpc_events. Only leads go to the intake crew.
LEAD_EVENTS = {"lead_intake", "client_created", None}
BUSINESS_EVENTS = {"payment_success", "appointment_booked", "invoice_created", "task_created"}
TEST_MARKERS = ("test", "zz-")


def _looks_like_test(payload):
    name = (payload.get("client_name") or "").lower()
    email = (payload.get("client_email") or payload.get("email") or "").lower()
    return payload.get("is_test") is True or any(m in name or m in email for m in TEST_MARKERS)


def route_business_event(payload):
    """True = handled here (payment/booking). False = it's a lead, run the normal intake."""
    event = payload.get("event")
    if event in LEAD_EVENTS or event not in BUSINESS_EVENTS:
        return False

    is_test = _looks_like_test(payload)
    supabase.table("dpc_events").insert({
        "event_type": event,
        "source": payload.get("source", "hub"),
        "client_name": payload.get("client_name"),
        "client_email": clean_email(payload.get("client_email") or payload.get("email")),
        "package": payload.get("package"),
        "amount": payload.get("amount"),
        "occurred_at": payload.get("occurred_at") or datetime.now(timezone.utc).isoformat(),
        "is_test": is_test,
        "payload": payload,
    }).execute()

    amount = payload.get("amount")
    msg = f"{event}: {payload.get('client_name') or 'unknown'}"
    if amount is not None:
        msg += f" (${amount})"
    log_event(
        "Siryandorin" if event == "payment_success" else "Diamond",
        event.upper(),
        msg + (" [TEST]" if is_test else ""),
        "OK"
    )
    return True


# ============================================================
# 3. ROUTE GUARDS
# ============================================================

def require_secret(f):
    """Accepts the secret as header X-DPC-Secret or query param ?key= (for Resend)."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not DPC_WEBHOOK_SECRET:
            print("❌ DPC_WEBHOOK_SECRET not set — rejecting request.", flush=True)
            return jsonify({"status": "error", "message": "Server auth not configured"}), 500
        provided = request.headers.get("X-DPC-Secret") or request.args.get("key") or ""
        if not hmac.compare_digest(provided.encode("utf-8"), DPC_WEBHOOK_SECRET.encode("utf-8")):
            return jsonify({"status": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


def require_db(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not supabase:
            return jsonify({"status": "error", "message": "Database not connected"}), 503
        return f(*args, **kwargs)
    return wrapper


def verify_shopify(f):
    """Verifies Shopify's HMAC signature against the raw request body."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not SHOPIFY_WEBHOOK_SECRET:
            print("❌ SHOPIFY_WEBHOOK_SECRET not set — rejecting request.", flush=True)
            return jsonify({"status": "error", "message": "Shopify auth not configured"}), 500
        header_hmac = request.headers.get("X-Shopify-Hmac-Sha256", "")
        digest = hmac.new(
            SHOPIFY_WEBHOOK_SECRET.encode("utf-8"),
            request.get_data(),
            hashlib.sha256
        ).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        if not hmac.compare_digest(header_hmac.encode("utf-8"), expected.encode("utf-8")):
            return jsonify({"status": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# ============================================================
# 4. TOOLS
# ============================================================

class SupabaseLogInput(BaseModel):
    message: str = Field(..., description="A short summary of what you just did or found.")
    action_type: str = Field(..., description="Category of action: e.g., 'READ_VAULT', 'SYSTEM_BOOT', 'CLIENT_INTAKE'")
    status: str = Field(default="INFO", description="Status: 'INFO', 'OK', or 'ERROR'")
    agent_name: str = Field(default="Diamond", description="The agent logging this: Diamond, Anansi, Kael, Siryandorin, Aurixen")

class SupabaseLoggerTool(BaseTool):
    name: str = "Dashboard Broadcast Tool"
    description: str = "CRITICAL: Use this tool to log your final answers, actions, and findings so the human can see them on the Command Dashboard. Always set agent_name to YOUR guardian name."
    args_schema: type[BaseModel] = SupabaseLogInput

    def _run(self, message: str, action_type: str, status: str = "INFO", agent_name: str = "Diamond") -> str:
        if not supabase:
            return "Database connection unavailable."
        try:
            supabase.table("agent_logs").insert({
                "agent_name": agent_name,
                "action_type": action_type,
                "message": message,
                "status": status
            }).execute()
            return "Successfully broadcasted to the Command Dashboard."
        except Exception as e:
            return f"Failed to log to database: {str(e)}"

db_logger_tool = SupabaseLoggerTool()


class ClientIntakeInput(BaseModel):
    client_name: str = Field(..., description="Name of the client.")
    package_tier: str = Field(default="LEAD", description="Package bought. Default to 'LEAD' if none.")
    revenue_value: float = Field(default=0.0, description="Amount paid. Default to 0 if none.")
    email: str = Field(default="", description="Client email address if present in the payload. Empty string if none.")

class ClientIntakeTool(BaseTool):
    name: str = "Client Pipeline Tool"
    description: str = "CRITICAL: Use this tool to officially add a new client to the DPC CRM pipeline database when they make a purchase or submit an intake. Always include the email if the payload has one."
    args_schema: type[BaseModel] = ClientIntakeInput

    def _run(self, client_name: str, package_tier: str = "LEAD", revenue_value: float = 0.0, email: str = "") -> str:
        if not supabase:
            return "Database connection unavailable."
        try:
            row = {
                "client_name": client_name,
                "package_tier": (package_tier or "LEAD").upper(),
                "status": "INTAKE",
                "progress_percentage": 10,
                "revenue_value": revenue_value if revenue_value is not None else 0.0
            }
            normalized = clean_email(email)
            if normalized:
                row["email"] = normalized
            supabase.table("dpc_clients").insert(row).execute()
            return f"Successfully added {client_name} to the CRM."
        except Exception as e:
            return f"Failed to add client to CRM: {str(e)}"

pipeline_tool = ClientIntakeTool()


class SupabaseReadInput(BaseModel):
    table: str = Field(..., description="The Supabase table to read from.")
    limit: int = Field(default=50, description="Number of rows to retrieve.")
    order_by: str = Field(default="created_at", description="Column to order by.")

class SupabaseReaderTool(BaseTool):
    name: str = "Supabase Data Reader"
    description: str = "Read raw data from a Supabase table for analysis. Use this to fetch cathedral_engagement, shopify_orders, dpc_clients, or any other table."
    args_schema: type[BaseModel] = SupabaseReadInput

    def _run(self, table: str, limit: int = 50, order_by: str = "created_at") -> str:
        if not supabase:
            return "Database connection unavailable."
        try:
            result = supabase.table(table).select("*").order(order_by, desc=True).limit(limit).execute()
            if not result.data:
                return f"No data found in table '{table}'."
            return str(result.data)
        except Exception as e:
            return f"Error reading from {table}: {str(e)}"

supabase_reader_tool = SupabaseReaderTool()


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
        if not supabase:
            return "Database connection unavailable."
        try:
            supabase.table("guardian_metrics").insert({
                "guardian": guardian.lower(),
                "metric_key": metric_key,
                "metric_value": metric_value,
                "summary": summary,
                "session_date": utc_today().isoformat()
            }).execute()
            log_event(
                guardian.upper(),
                "GUARDIAN_SYNC",
                f"{guardian.title()} updated {metric_key}: {metric_value} — {summary[:80]}",
                "OK"
            )
            return f"Successfully wrote {guardian} metrics to Supabase."
        except Exception as e:
            return f"Failed to write Guardian metrics: {str(e)}"

guardian_metrics_tool = GuardianMetricsWriterTool()


# ============================================================
# 5. LLM
# ============================================================
# Model comes from Railway variable ANTHROPIC_MODEL (e.g. claude-sonnet-5-5).
# Swap models there, no code edit. CrewAI needs the "anthropic/" prefix, added here.
CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
if not CLAUDE_MODEL.startswith("anthropic/"):
    CLAUDE_MODEL = f"anthropic/{CLAUDE_MODEL}"

claude_llm = LLM(
    model=CLAUDE_MODEL,
    max_tokens=4096
)


# ============================================================
# 6. AGENTS
# ============================================================

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

# Anansi (analyst) — used in Guardian Sync. No email tool: Anansi never sends mail directly.
anansi_agent = Agent(
    role='Anansi — Guardian of Brand & Community',
    goal='Analyze community health and brand resonance across the DPC pipeline.',
    backstory=(
        "You are Anansi, the Guardian of Brand and Community within the Diamond Protocol. "
        "Your domain is the story, resonance, and relationships of DPC. "
        "You calculate Narrative Density and community health metrics."
    ),
    verbose=True,
    tools=[supabase_reader_tool, guardian_metrics_tool, db_logger_tool],
    llm=claude_llm
)

# Anansi (writer) — drafts emails only. Python does the sending.
anansi_writer = Agent(
    role='Anansi — Voice of DPC',
    goal='Draft personalized, culturally resonant emails to leads entering the DPC ecosystem.',
    backstory=(
        "You are Anansi, Guardian of Brand and Community and the voice of "
        "frontdesk@daplayerscollective.com. You write emails that speak directly to what each person "
        "actually said. Your tone is authoritative but welcoming. You never send generic templates. "
        "You only draft — you do not send."
    ),
    verbose=True,
    tools=[],
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
# 7. EMAIL DRAFTING (LLM drafts, Python sends)
# ============================================================

class EmailDraft(BaseModel):
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Plain-text email body.")


def extract_draft(result):
    draft = getattr(result, "pydantic", None)
    if draft is not None and getattr(draft, "subject", None) and getattr(draft, "body", None):
        return draft.subject.strip(), draft.body.strip()

    raw = getattr(result, "raw", None) or str(result)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        if data.get("subject") and data.get("body"):
            return str(data["subject"]).strip(), str(data["body"]).strip()

    raise ValueError("Anansi did not return a usable subject/body draft.")


def draft_email(instructions):
    task = Task(
        description=(
            f"{instructions}\n\n"
            "Return ONLY a JSON object with two fields: 'subject' and 'body'. "
            "The body is plain text. No markdown, no preamble."
        ),
        expected_output="A JSON object with 'subject' and 'body' fields.",
        agent=anansi_writer,
        output_pydantic=EmailDraft
    )
    result = Crew(agents=[anansi_writer], tasks=[task], process=Process.sequential).kickoff()
    return extract_draft(result)


# ============================================================
# 8. BACKGROUND JOBS
# ============================================================

def run_intake_crew(payload):
    try:
        analyze_task = Task(
            description=(
                f"Analyze this incoming webhook payload: {payload}. Identify the client name and email if present. "
                "If 'package' or 'amount' are null or missing, this is a free Lead. Set the package to 'LEAD' "
                "and amount to 0. Pass a summary to the next agent."
            ),
            expected_output="A short summary of the client, their email if any, and their purchase or lead status.",
            agent=angel_orchestrator
        )
        execute_task = Task(
            description=(
                "Take Angel's summary. First, use the Client Pipeline Tool to add the client to the database, "
                "including their email if one was found. Ensure you handle missing packages/amounts correctly. "
                "Then, use the Dashboard Broadcast Tool to log 'New Intake: [Client Name] starting [Package]' "
                "with action_type 'CLIENT_INTAKE' and status 'OK'."
            ),
            expected_output="Confirmation that the client was added to the CRM and the broadcast was logged.",
            agent=diamond_agent
        )
        Crew(
            agents=[angel_orchestrator, diamond_agent],
            tasks=[analyze_task, execute_task],
            process=Process.sequential
        ).kickoff()
    except Exception as e:
        print("❌ INTAKE CREW FAILED:", flush=True)
        traceback.print_exc()
        log_event("Angel", "CLIENT_INTAKE", f"❌ Intake crew failed: {str(e)[:150]}", "ERROR")


def run_anansi_outreach(lead_data):
    client_id = lead_data.get("id")
    client_name = lead_data.get("client_name") or "Unknown Client"
    try:
        intent = str(lead_data.get("intent") or "").lower()
        budget = parse_money(lead_data.get("budget"))
        message = lead_data.get("message") or "No specific message provided."
        recipient = clean_email(lead_data.get("email"))

        # High ticket → human handles it
        if budget >= 3000 or "founder" in intent:
            print(f"🚨 HIGH TICKET LEAD: {client_name}", flush=True)
            log_event("Angel", "HIGH_TICKET_ALERT", f"🚨 High-Ticket Lead: {client_name} (${budget:,.0f})", "INFO")
            supabase.table("dpc_clients").update({"status": "Needs Manual Outreach"}).eq("id", client_id).execute()
            return

        # No email → can't contact, don't pretend we did
        if not recipient:
            log_event("Anansi", "LEAD_OUTREACH", f"⚠️ {client_name} has no email — needs manual outreach.", "WARN")
            supabase.table("dpc_clients").update({"status": "Needs Email - Manual"}).eq("id", client_id).execute()
            return

        log_event("Anansi", "LEAD_OUTREACH", f"Anansi drafting outreach for {client_name} ({recipient})", "INFO")

        subject, body = draft_email(f"""
A new creative just joined via the Sovereignty Hub.
Name: {client_name}
Intent/Interest: {intent}
Their Message: "{message}"

Write a personalized welcome email that speaks directly to what they said.
Identify their intent and pain point and reference their actual message.
Sign as Anansi, Guardian of Brand & Community, Da Players Collective.
""")

        ok, info = send_email(recipient, subject, body)

        if not ok:
            print(f"❌ OUTREACH SEND FAILED for {client_name}: {info}", flush=True)
            log_event("Anansi", "LEAD_OUTREACH", f"❌ Welcome email failed for {client_name}: {info}", "ERROR")
            supabase.table("dpc_clients").update({"status": "Outreach Failed"}).eq("id", client_id).execute()
            return

        follow_up = (utc_today() + timedelta(days=2)).isoformat()
        supabase.table("dpc_clients").update({
            "status": "Contacted",
            "nurture_stage": 1,
            "follow_up_date": follow_up
        }).eq("id", client_id).execute()

        log_event("Anansi", "LEAD_OUTREACH", f"✅ Welcome sent to {client_name} (Resend id: {info}). First nurture: {follow_up}", "OK")
        print(f"✅ LEAD PROCESSED: {client_name} → Contacted", flush=True)

    except Exception as e:
        print(f"❌ BACKGROUND LEAD PROCESSING FAILED: {str(e)}", flush=True)
        traceback.print_exc()
        log_event("Anansi", "LEAD_OUTREACH", f"❌ Outreach failed for {client_name}: {str(e)[:100]}", "ERROR")
        try:
            supabase.table("dpc_clients").update({"status": "Outreach Failed"}).eq("id", client_id).execute()
        except Exception:
            pass


NURTURE_STAGE_CONTEXT = {
    1: "Send a value-packed insight directly related to their original message. Teach something useful. No pitch.",
    2: "Share a brief story of someone with a similar goal who found success through DPC's framework. Make it real and specific.",
    3: "Introduce what working with DPC actually looks like. One clear offer. One clear next step.",
    4: "This is the decision email. Create urgency around their original goal. Ask them directly if they're ready to move."
}

# Days to wait after sending stage N before stage N+1. None = sequence complete.
NURTURE_STAGE_DELAYS = {1: 2, 2: 3, 3: 4, 4: None}


def reschedule_tomorrow(lead_id):
    try:
        tomorrow = (utc_today() + timedelta(days=1)).isoformat()
        supabase.table("dpc_clients").update({"follow_up_date": tomorrow}).eq("id", lead_id).execute()
    except Exception as e:
        print(f"❌ Failed to reschedule lead {lead_id}: {e}", flush=True)


def run_anansi_nurture(lead_data):
    lead_id = lead_data.get("id")
    client_name = lead_data.get("client_name") or "Unknown"
    try:
        recipient = clean_email(lead_data.get("email"))
        message = lead_data.get("message") or ""
        stage = int(lead_data.get("nurture_stage") or 1)

        if stage > 4:
            return  # already complete; route already cleared follow_up_date

        if not recipient:
            supabase.table("dpc_clients").update({"status": "Needs Email - Manual"}).eq("id", lead_id).execute()
            log_event("Anansi", "NURTURE_SEQUENCE", f"⚠️ {client_name} has no email — nurture stopped.", "WARN")
            return

        log_event("Anansi", "NURTURE_SEQUENCE", f"Anansi drafting stage {stage} nurture for {client_name}", "INFO")

        subject, body = draft_email(f"""
You are continuing a nurture sequence with an existing lead.
Name: {client_name}
Original Message: "{message}"
Sequence Stage: {stage} of 4

Your instruction for this stage:
{NURTURE_STAGE_CONTEXT[stage]}

Rules:
- Reference their original message to show you remember them
- Do NOT repeat the welcome. They already got that.
- Match the stage instruction exactly
- Keep it under 200 words
- Sign as Anansi, Guardian of Brand & Community
""")

        ok, info = send_email(recipient, subject, body)

        if not ok:
            log_event("Anansi", "NURTURE_SEQUENCE", f"❌ Stage {stage} failed for {client_name}: {info}. Retrying tomorrow.", "ERROR")
            reschedule_tomorrow(lead_id)
            return

        # If they replied while we were drafting, don't restart the sequence
        current = supabase.table("dpc_clients").select("status").eq("id", lead_id).execute()
        if current.data and current.data[0].get("status") == "Replied - Action Required":
            log_event("Anansi", "NURTURE_SEQUENCE", f"Stage {stage} sent to {client_name}, but they already replied — sequence stays halted.", "WARN")
            return

        delay = NURTURE_STAGE_DELAYS.get(stage)
        if delay:
            next_date = (utc_today() + timedelta(days=delay)).isoformat()
            supabase.table("dpc_clients").update({
                "nurture_stage": stage + 1,
                "follow_up_date": next_date
            }).eq("id", lead_id).execute()
        else:
            next_date = None
            supabase.table("dpc_clients").update({
                "nurture_stage": 5,
                "follow_up_date": None,
                "status": "Sequence Complete"
            }).eq("id", lead_id).execute()

        log_event("Anansi", "NURTURE_SEQUENCE",
                  f"✅ Stage {stage} sent to {client_name} (Resend id: {info}). Next: {next_date or 'Sequence complete'}", "OK")

    except Exception as e:
        print(f"❌ NURTURE FAILED: {str(e)}", flush=True)
        traceback.print_exc()
        log_event("Anansi", "NURTURE_SEQUENCE", f"❌ Nurture crashed for {client_name}: {str(e)[:100]}. Retrying tomorrow.", "ERROR")
        reschedule_tomorrow(lead_id)


def run_guardian_crew():
    try:
        print("⚡ [GUARDIAN SYNC INITIATED]", flush=True)
        log_event("Angel", "GUARDIAN_SYNC", "⚡ Guardian sync started", "INFO")

        kael_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch the latest 100 rows from 'cathedral_engagement'. "
                "Analyze the data: count total events, calculate average value, identify the most active rank levels, "
                "and assess consistency of engagement over time. "
                "Calculate an Energy Score from 0-100 based on volume and value of engagement. "
                "Then use the Guardian Metrics Writer to save: metric_key='energy_score', your calculated score, "
                "and a 1-2 sentence summary of community energy. "
                "Finally use the Dashboard Broadcast Tool to log your completion with action_type 'GUARDIAN_SYNC' "
                "and agent_name 'Kael'."
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
                "Log completion with the Dashboard Broadcast Tool with action_type 'GUARDIAN_SYNC' "
                "and agent_name 'Siryandorin'."
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
                "Log completion with the Dashboard Broadcast Tool with action_type 'GUARDIAN_SYNC' "
                "and agent_name 'Anansi'."
            ),
            expected_output="Confirmation that narrative_density has been written to guardian_metrics.",
            agent=anansi_agent
        )

        aurixen_task = Task(
            description=(
                "Use the Supabase Data Reader to fetch the latest rows from 'guardian_metrics' to read what Kael, "
                "Siryandorin, and Anansi have just written. Also read 'dpc_clients' and 'shopify_orders' for raw context. "
                "Cross-reference all domains: identify conflicts, risks, and strategic opportunities. "
                "Calculate a Risk Index from 0-100 (higher = more risk) and a Strategic Opportunity score from 0-100. "
                "Use the Guardian Metrics Writer twice: once for metric_key='risk_index' and once for metric_key='opportunity_score'. "
                "Log completion with the Dashboard Broadcast Tool with action_type 'GUARDIAN_SYNC' "
                "and agent_name 'Aurixen'."
            ),
            expected_output="Confirmation that risk_index and opportunity_score have been written to guardian_metrics.",
            agent=aurixen_agent
        )

        Crew(
            agents=[kael_agent, siryandorin_agent, anansi_agent, aurixen_agent],
            tasks=[kael_task, siryandorin_task, anansi_task, aurixen_task],
            process=Process.sequential
        ).kickoff()

        log_event("Angel", "GUARDIAN_SYNC", "✅ Guardian sync complete", "OK")

    except Exception as e:
        print("❌ GUARDIAN SYNC FAILED:", flush=True)
        traceback.print_exc()
        log_event("Angel", "GUARDIAN_SYNC", f"❌ Guardian sync failed: {str(e)[:150]}", "ERROR")


# ============================================================
# 9. ROUTES
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "db_connected": bool(supabase)}), 200


@app.route('/intake-webhook', methods=['POST'])
@require_secret
@require_db
def handle_intake():
    payload = request.get_json(silent=True) or {}
    print(f"📡 [WEBHOOK RECEIVED]: {payload}", flush=True)

    # Payments/bookings go to dpc_events and stop here. Leads fall through to the intake crew.
    try:
        if route_business_event(payload):
            return jsonify({"status": "success", "routed": payload.get("event")}), 200
    except Exception as e:
        traceback.print_exc()
        log_event("Diamond", "EVENT_ROUTING", f"❌ Event routing failed: {str(e)[:150]}", "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 500

    run_in_background(run_intake_crew, payload)
    return jsonify({"status": "accepted", "message": "Intake received, DPC Swarm dispatched"}), 202


@app.route('/cathedral-webhook', methods=['POST'])
@require_secret
@require_db
def handle_cathedral():
    try:
        data = request.get_json(silent=True) or {}
        print(f"🏛️ [CATHEDRAL EVENT]: {data}", flush=True)

        supabase.table("cathedral_engagement").insert({
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "rank_level": data.get("rank_level"),
            "value": data.get("value"),
            "event_type": data.get("event", "rank_up"),
            "test_mode": data.get("test", False)
        }).execute()

        log_event("Cathedral", "RANK_UP",
                  f"{data.get('username')} ranked up to {data.get('rank_level')} (Value: {data.get('value')})", "OK")

        return jsonify({"status": "success", "message": "Cathedral engagement tracked"}), 200
    except Exception as e:
        print(f"❌ CATHEDRAL WEBHOOK ERROR: {str(e)}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# Left unauthenticated on purpose: aggregate rank stats only, may be called from the dashboard frontend.
@app.route('/api/engagement', methods=['GET'])
@require_db
def get_engagement_data():
    try:
        result = supabase.table("cathedral_engagement").select("*").order("created_at", desc=True).limit(100).execute()

        rank_stats = {}
        for row in result.data:
            rank = row.get('rank_level') or 'unknown'
            if rank not in rank_stats:
                rank_stats[rank] = {'count': 0, 'total_value': 0}
            rank_stats[rank]['count'] += 1
            rank_stats[rank]['total_value'] += float(row.get('value') or 0)

        cohorts = [
            {
                "name": str(rank_name).title(),
                "completion": min(100, int(stats['total_value'] / 10)),
                "users": stats['count']
            }
            for rank_name, stats in rank_stats.items()
        ]
        cohorts.sort(key=lambda x: x['completion'], reverse=True)

        coupons_result = supabase.table("agent_logs").select("message, created_at").eq("action_type", "RANK_UP").order("created_at", desc=True).limit(5).execute()
        coupons = [
            {
                "code": f"RANK-{row['created_at'][:10].replace('-', '')}",
                "reward": (row.get('message') or '')[:50],
                "triggered": row['created_at']
            }
            for row in coupons_result.data
        ]

        return jsonify({"cohorts": cohorts, "coupons": coupons}), 200

    except Exception as e:
        print(f"❌ ENGAGEMENT API ERROR: {str(e)}", flush=True)
        return jsonify({"cohorts": [], "coupons": []}), 200


@app.route('/shopify-webhook', methods=['POST'])
@verify_shopify
@require_db
def handle_shopify():
    try:
        data = request.get_json(silent=True) or {}
        print(f"🛒 [SHOPIFY ORDER]: {data.get('id')}", flush=True)

        customer = data.get("customer") or {}
        customer_name = " ".join(
            part for part in [customer.get("first_name"), customer.get("last_name")] if part
        ).strip() or data.get("email") or "Guest"

        # Upsert so Shopify retries don't double-count revenue (requires UNIQUE on order_id)
        supabase.table("shopify_orders").upsert({
            "order_id": data.get("id"),
            "customer_name": customer_name,
            "total_price": data.get("total_price"),
            "created_at": data.get("created_at")
        }, on_conflict="order_id").execute()

        log_event("Shopify", "ORDER_RECEIVED", f"Order from {customer_name}: ${data.get('total_price')}", "OK")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"❌ SHOPIFY WEBHOOK ERROR: {str(e)}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/dpc-client-webhook', methods=['POST'])
@require_secret
@require_db
def handle_new_client():
    try:
        payload = request.get_json(silent=True) or {}
        print(f"🎯 [LEAD WEBHOOK RECEIVED]: {payload.get('type')} on {payload.get('table')}", flush=True)

        if payload.get("table") != "dpc_clients" or payload.get("type") != "INSERT":
            return jsonify({"status": "ignored", "reason": "Not a new dpc_client insert"}), 200

        lead_data = payload.get("record") or {}
        run_in_background(run_anansi_outreach, lead_data)

        return jsonify({"status": "success", "message": "Lead received, Anansi dispatched"}), 200

    except Exception as e:
        print("❌ ERROR IN DPC CLIENT WEBHOOK:", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/nurture-sequence', methods=['POST'])
@require_secret
@require_db
def run_nurture_sequence():
    """
    Called daily by Railway cron.
    Picks up every lead due today OR overdue (missed cron days), claims it, then fires Anansi.
    """
    try:
        today = utc_today().isoformat()

        due = supabase.table("dpc_clients")\
            .select("*")\
            .lte("follow_up_date", today)\
            .eq("package_tier", "LEAD")\
            .execute()

        if not due.data:
            return jsonify({"status": "success", "message": "No leads due today"}), 200

        for lead in due.data:
            # Claim the lead so a second cron hit can't double-send
            supabase.table("dpc_clients").update({"follow_up_date": None}).eq("id", lead["id"]).execute()
            run_in_background(run_anansi_nurture, lead)

        return jsonify({"status": "success", "leads_processing": len(due.data)}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/inbound-reply', methods=['POST'])
@require_secret
@require_db
def handle_inbound_reply():
    """
    Receives inbound email webhooks from Resend (secret passed as ?key=).
    Finds the matching lead and halts their nurture sequence.
    """
    try:
        payload = request.get_json(silent=True) or {}
        print(f"📥 [INBOUND REPLY RECEIVED]: {payload.get('type')}", flush=True)

        if payload.get("type") != "email.received":
            return jsonify({"status": "ignored", "message": "Not a received email event"}), 200

        event_data = payload.get("data") or {}
        sender_email = clean_email(event_data.get("from"))

        if not sender_email:
            return jsonify({"status": "error", "message": "No sender found"}), 400

        lead = supabase.table("dpc_clients")\
            .select("id, client_name, status")\
            .ilike("email", sender_email)\
            .execute()

        if not lead.data:
            return jsonify({"status": "ignored", "message": "Sender not in pipeline"}), 200

        lead_id = lead.data[0]["id"]
        client_name = lead.data[0]["client_name"]

        # THE KILL SWITCH
        supabase.table("dpc_clients").update({
            "follow_up_date": None,
            "status": "Replied - Action Required"
        }).eq("id", lead_id).execute()

        log_event("Anansi", "SEQUENCE_HALTED", f"⏸️ Sequence stopped for {client_name}. They replied to Anansi.", "WARN")

        return jsonify({"status": "success", "message": "Sequence halted successfully"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/guardian-sync', methods=['POST'])
@require_secret
@require_db
def run_guardian_sync():
    """Kicks off the Guardian crew in the background. Watch agent_logs for completion."""
    run_in_background(run_guardian_crew)
    return jsonify({"status": "accepted", "message": "Guardian sync started. Watch the dashboard feed for completion."}), 202


@app.route('/api/guardian-metrics', methods=['GET'])
@require_secret
@require_db
def get_guardian_metrics():
    """Latest metrics for all four Guardians. Used by Flowise before a Council Session."""
    try:
        result = supabase.table("guardian_metrics").select("*").order("created_at", desc=True).limit(50).execute()

        metrics = {}
        for row in result.data:
            guardian = row.get("guardian")
            metrics.setdefault(guardian, []).append({
                "metric_key": row.get("metric_key"),
                "metric_value": row.get("metric_value"),
                "summary": row.get("summary"),
                "session_date": row.get("session_date")
            })

        return jsonify({"status": "success", "guardian_metrics": metrics}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/council-briefing', methods=['GET'])
@require_secret
@require_db
def get_council_briefing():
    """
    Snapshot for Flowise at the start of every Council Session.
    NOTE: counts here are from recent rows only, not all-time totals. Real totals = Pass 2.
    """
    try:
        clients = supabase.table("dpc_clients").select("*").order("created_at", desc=True).limit(10).execute()
        leads = [c for c in clients.data if c.get("package_tier") == "LEAD"]
        paying = [c for c in clients.data if c.get("package_tier") != "LEAD"]

        orders = supabase.table("shopify_orders").select("*").order("created_at", desc=True).limit(10).execute()
        total_revenue = sum(parse_money(o.get("total_price")) for o in orders.data)

        engagement = supabase.table("cathedral_engagement").select("*").order("created_at", desc=True).limit(20).execute()

        gm = supabase.table("guardian_metrics").select("*").order("created_at", desc=True).limit(20).execute()
        guardian_summary = {}
        for row in gm.data:
            g = row.get("guardian")
            guardian_summary.setdefault(g, {})
            # Keep only the newest value per metric
            if row.get("metric_key") not in guardian_summary[g]:
                guardian_summary[g][row.get("metric_key")] = {
                    "value": row.get("metric_value"),
                    "summary": row.get("summary")
                }

        briefing = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": {
                "recent_leads": len(leads),
                "recent_paying_clients": len(paying),
                "recent_intakes": [c.get("client_name") for c in clients.data[:5]]
            },
            "revenue": {
                "recent_orders": len(orders.data),
                "recent_revenue": round(total_revenue, 2)
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
# 10. BOOT (local only — Railway should run gunicorn)
# ============================================================
if __name__ == '__main__':
    print("🎧 Angel Orchestrator is online and listening...", flush=True)
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)