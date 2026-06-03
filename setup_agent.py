import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

AGENT_ID = "agent_011CZxrRqFmxnvr87F7KtA22"
ENV_ID = "env_01JbZWBFVXC7YQszXFSypfM7"

session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENV_ID,
    title="My New Task",
    betas=["managed-agents-2026-04-01"],
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

client.beta.sessions.events.send(
    session_id=session.id,
    events=[{
        "type": "user.message",
        "content": [{"type": "text", "text": f"""You have access to my Supabase database:
URL: {SUPABASE_URL}
Service Key: {SUPABASE_KEY}

NOW DO THIS: <your actual task here>"""}],
    }],
    betas=["managed-agents-2026-04-01"],
)

for event in client.beta.sessions.events.stream(
    session_id=session.id,
    betas=["managed-agents-2026-04-01"],
):
    if event.type == "agent.message":
        for block in event.content:
            if block.type == "text":
                print(block.text)
    elif event.type == "session.status_idle":
        print("\n✅ Done.")
        break
    elif event.type == "session.error":
        print(f"\n❌ Error: {event}")
        break