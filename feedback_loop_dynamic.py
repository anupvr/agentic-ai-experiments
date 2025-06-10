import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API key
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Generate initial explanation
initial_prompt = PromptTemplate.from_template(
    "Write a short paragraph explaining what agentic AI is."
)
summary_chain = LLMChain(llm=llm, prompt=initial_prompt)

# Step 2: Review output
review_prompt = PromptTemplate.from_template(
    "Review this explanation and suggest improvements. "
    "If it's already clear, say 'No improvement needed.'\n\n\"{output}\""
)
review_chain = LLMChain(llm=llm, prompt=review_prompt)

# Step 3: Refine based on feedback
refine_prompt = PromptTemplate.from_template(
    "Improve the explanation based on the feedback.\n\n"
    "Explanation: \"{output}\"\n"
    "Feedback: \"{feedback}\"\n\n"
    "Improved version:"
)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# Main loop with dynamic depth
if __name__ == "__main__":
    output = summary_chain.run({})
    print("\n🔹 Initial Draft:")
    print(output)

    max_rounds = 3
    for i in range(max_rounds):
        print(f"\n🔍 Review Round {i + 1}")
        feedback = review_chain.run({"output": output})
        print("Feedback:", feedback.strip())

        if "no improvement" in feedback.lower():
            print("✅ Feedback indicates the explanation is sufficient.")
            break

        output = refine_chain.run({"output": output, "feedback": feedback})
        print("🔁 Updated Explanation:", output.strip())
    else:
        print("\n⚠️ Max refinement attempts reached. Final output used.")

    print("\n✅ Final Output:")
    print(output.strip())