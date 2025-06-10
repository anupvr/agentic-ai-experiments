from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)
#llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------- Agents ----------------
manager = Agent(
    role="Manager",
    goal="Delegate tasks based on the user's request",
    backstory="An experienced coordinator who analyzes needs and assigns the best agent.",
    llm=llm,
    verbose=True
)

researcher = Agent(
    role="Researcher",
    goal="Research the topic and find accurate information",
    backstory="Expert at finding factual data on any subject.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a summary or report based on research",
    backstory="Writes concise and engaging summaries based on facts.",
    llm=llm,
    verbose=True
)

# ---------------- Tasks ----------------
initial_task = Task(
    description="Decide whether the user needs research or content writing on the topic: 'Impact of climate change on agriculture'. Then assign accordingly.",
    agent=manager
)

# ↓ These agents are used only if delegated
research_task = Task(
    description="Research the effects of climate change on agriculture and summarize in bullet points.",
    agent=researcher
)

write_task = Task(
    description="Write a 2-paragraph article about the impact of climate change on farming based on available facts.",
    agent=writer
)

# ---------------- Crew (Hierarchical) ----------------
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[initial_task, research_task, write_task],
    process="hierarchical",
    verbose=True
)

# ---------------- Run ----------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n✅ Final Output:\n", result)