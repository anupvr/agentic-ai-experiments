#pip install crewai langchain openai python-dotenv
import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# Setup your LLM
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0,api_key = openai_key)

# ---------------- Agents ----------------
researcher = Agent(
    role='Researcher',
    goal='Find accurate and relevant information on the given topic',
    backstory='An expert researcher with access to a large knowledge base.',
    verbose=True,
    llm=llm
)

critic = Agent(
    role='Critic',
    goal='Review the researcher’s notes and suggest improvements',
    backstory='A critical thinker who ensures high-quality and unbiased information.',
    verbose=True,
    llm=llm
)

writer = Agent(
    role='Writer',
    goal='Generate a well-written, clear report based on the reviewed content',
    backstory='A professional writer skilled at simplifying technical info.',
    verbose=True,
    llm=llm
)

# ---------------- Tasks (with expected_output) ----------------
task1 = Task(
    description="Research the topic: 'Impact of AI in Education' and provide a bullet point summary.",
    expected_output="A concise bullet list with 5–7 key points on AI's impact in education.",
    agent=researcher
)

task2 = Task(
    description="Review the research notes and rewrite them with clarity and logical flow.",
    expected_output="A refined version of the bullet list with improved structure and clarity.",
    agent=critic
)

task3 = Task(
    description="Write a short report (2–3 paragraphs) using the refined summary.",
    expected_output="A clear, coherent article about AI in education using the previous summary.",
    agent=writer
)
# ---------------- Crew ----------------
crew = Crew(
    agents=[researcher, critic, writer],
    tasks=[task1, task2, task3],
    verbose=True,
    process='sequential'  # agents work in order
)

# ---------------- Run It ----------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n📝 Final Report:\n", result)