import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv

# Fix Windows UTF-8 encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul')

# Load environment variables from .env file (SECURE!)
load_dotenv()

# Ensure keys are set
if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("SERPER_API_KEY"):
    raise ValueError("API keys must be set in .env file")

scrape_tool = ScrapeWebsiteTool()

print("Starting the Diamond Protocol Swarm Audit...")

# 1. Kael (The Void Architect)
kael = Agent(
    role='Void Architect',
    goal='Audit backend infrastructure and funnel structure of Da Players Collective',
    backstory='You are a cold, precise systems architect. You look for broken links, inefficient user flows, and structural weaknesses in funnels.',
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm="claude-sonnet-4-20250514"  #  String format is correct!
)

# 2. Siryandorin (The Golden Dragon)
siryandorin = Agent(
    role='Golden Dragon',
    goal='Maximize high-ticket conversions and pricing strategy for the Sovereign Hub',
    backstory='You are obsessed with wealth accumulation, pricing psychology, and high-ticket sales.',
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm="claude-sonnet-4-20250514"
)

# 3. Aurixen (The Quantum Weaver)
aurixen = Agent(
    role='Quantum Weaver',
    goal='Identify viral traffic loops and speed-to-market strategies',
    backstory='You are kinetic and non-linear. You look for ways to engineer virality.',
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm="claude-sonnet-4-20250514"
)

# 4. Anansi (The Digital Spider)
anansi = Agent(
    role='Digital Spider',
    goal='Audit brand narrative, esoteric storytelling, and cultural resonance',
    backstory='You are a trickster and storyteller. You ensure the brand feels mythological.',
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm="claude-sonnet-4-20250514"
)

# 4 Tasks (One Per Agent) 
task_kael = Task(
    description='Audit backend infrastructure and funnel structure.',
    expected_output='Infrastructure audit report with bottlenecks identified.',
    agent=kael
)

task_siryandorin = Task(
    description='Analyze pricing strategy for Sovereign Hub.',
    expected_output='Pricing strategy recommendations.',
    agent=siryandorin
)

task_aurixen = Task(
    description='Identify viral traffic loops.',
    expected_output='Traffic acceleration plan.',
    agent=aurixen
)

task_anansi = Task(
    description='Audit brand narrative and cultural resonance.',
    expected_output='Brand narrative recommendations.',
    agent=anansi
)

# The Crew
diamond_swarm = Crew(
    agents=[kael, siryandorin, aurixen, anansi],
    tasks=[task_kael, task_siryandorin, task_aurixen, task_anansi],  #  All 4 tasks
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = diamond_swarm.kickoff()
    print("\n\n########################")
    print("## Sovereign Blueprint Final Output:")  #  No emoji
    print("########################\n")
    print(result)