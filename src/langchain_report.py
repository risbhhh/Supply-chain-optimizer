"""
LangChain-based reporting module.
Turns optimization results into an executive-style summary.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def generate_report(results: dict):
    """
    Generate a textual executive report from optimization results.
    
    Args:
        results (dict): output of optimizer (orders, inventory, shortages, objective)
    
    Returns:
        str: formatted executive summary
    """
    template = """
You are a supply chain consultant.
Summarize the optimization results for a manager in clear, non-technical language.

Optimization results:
{results}

Write 2-3 short paragraphs with:
- A summary of the decisions (e.g., order plan, inventory levels, shortages)
- Key risks and trade-offs
- 3 actionable recommendations
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["results"],
    )

    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    chain = LLMChain(prompt=prompt, llm=llm)

    return chain.run(results=str(results))


if __name__ == "__main__":
    fake_results = {
        "orders": [50, 0, 20, 0, 30],
        "inventory": [70, 40, 30, 10, 5],
        "shortages": [0, 0, 0, 5, 10],
        "objective": 560.0,
    }
    print(generate_report(fake_results))
