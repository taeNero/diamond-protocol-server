import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create LangChain LLM object explicitly
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Agent with LangChain LLM
agent = Agent(
    role='Tester',
    goal='Test if CrewAI works with OpenAI',
    backstory='You are a test agent.',
    verbose=True,
    llm=llm  # ← Pass LangChain object instead of string
)

task = Task(
    description='Say hello in one sentence.',
    expected_output='A greeting.',
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

if __name__ == "__main__":
    print("Testing CrewAI with LangChain...")
    result = crew.kickoff()
    print(result)
