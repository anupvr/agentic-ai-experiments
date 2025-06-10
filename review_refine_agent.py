import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API keys from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=openai_key)

# Step 1: Initial explanation prompt
initial_prompt = PromptTemplate.from_template(
    "Write a short paragraph explaining what agentic AI is."
)
summary_chain = LLMChain(llm=llm, prompt=initial_prompt)

# Step 2: Review the explanation for improvements
review_prompt = PromptTemplate.from_template(
    "Review the following explanation and list at least one improvement:\n\n\"{output}\""
)
review_chain = LLMChain(llm=llm, prompt=review_prompt)

# Step 3: Refine based on feedback
refine_prompt = PromptTemplate.from_template(
    "Improve this explanation based on the following feedback.\n\nExplanation: \"{output}\"\nFeedback: \"{feedback}\"\n\nImproved version:"
)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# Run the self-review + refine cycle
if __name__ == "__main__":
    print("\n🔹 Step 1: Generate Initial Explanation")
    original = summary_chain.run({})
    print("Initial:", original)

    print("\n🔍 Step 2: Review Output")
    feedback = review_chain.run({"output": original})
    print("Feedback:", feedback)

    print("\n🔁 Step 3: Refine Output")
    improved = refine_chain.run({"output": original, "feedback": feedback})
    print("Final Output:", improved)