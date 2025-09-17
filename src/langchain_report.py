"""Builds prompts and calls an LLM (via LangChain) to generate human-friendly explanation of the optimization results.
You must set OPENAI_API_KEY in your environment."
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def generate_report(optimization_summary: dict, top_insights: list):
    template = """
You are a supply-chain analyst. Given the optimization summary below, produce a short executive report (2-4 paragraphs) and bullet-point actionable insights.

Optimization summary:
{summary}

Top numerical insights:
{insights}

Write a concise executive summary and recommend 3 actions.
"""
    prompt = PromptTemplate(template=template, input_variables=["summary", "insights"])
    llm = OpenAI(temperature=0.2, max_tokens=400)
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run({"summary": str(optimization_summary), "insights": '\n'.join(top_insights)})
    return out


if __name__ == '__main__':
    print(generate_report({'order_qty': 120, 'objective': 345.6}, ['Expected inventory 80 units', 'Order qty 120']))
