from crewai import Agent, Task, Crew, LLM, Process
from crewai.project import CrewBase, agent, crew
from crewai.tasks.task_output import TaskOutput
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from exa_py import Exa
from typing import Type, Union
import requests
import logging
from scripts.regsetup import description

# run file:
# python ai_agents.py

load_dotenv()

# Set up a logger for debugging and production logging.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- LLM Configuration ---------------------
llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------- EXA Answer Tool ---------------------
"""
Future improvements: Add tools for web scraping, data parsing, or citation management
to further enhance the agent's research capabilities.
"""
class EXAAnswerToolSchema(BaseModel):
    query: str = Field(description='The query you want to ask the EXA.')

class EXXAnswerTool(BaseTool):
    name: str = "EXA Answer Tool"
    description: str = "A Tool to answer user query using EXA"
    args_schema: Type[BaseModel] = EXAAnswerToolSchema
    answer_url: str = "https://api.exa.ai/answer"
    headers: dict = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": os.getenv("EXA_API_KEY"),
    }

    def _run(self, query: Union[str, dict]) -> str:
        # Handle case where query is provided as a dict (with field metadata)
        if isinstance(query, dict):
            # Try to extract the actual query string from the dict.
            query_value = query.get("description")
            if not isinstance(query_value, str):
                query_value = str(query)
            query = query_value

        # Debugging: Ensure query is a string
        print("DEBUG: query type:", type(query))
        print("DEBUG: query content:", query)

        try:
            response = requests.post(
                self.answer_url,
                json={"query": query},
                headers=self.headers,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_error:
            print(f"HTTP error occurred: {http_error}")
            print(f"Error response: {response.content}")
            raise
        except Exception as error:
            print(f"An error occurred: {error}")
            raise

        response_data = response.json()
        answer = response_data.get("answer")
        citations = response_data.get("citation", [])
        output = f"Answer: {answer}\n\n"
        if citations:
            output += "Citations:\n"
            for citation in citations:
                output += f"- {citation['title']} ({citation['url']})\n"
        return output

# ---------- Callback Function ---------------------
def callback_function(output: TaskOutput):
    # Using a formatted string for clearer output
    print(f"""
Agent: {output.agent}
Task: {output.description}
Task Summary: {output.summary}
Output: {output.raw}
""")

# ---------- Research Agent Configuration ---------------------
research_agent = Agent(
    role="Research Analyst",
    goal=("Conduct thorough research on the given {topic}, providing detailed analysis, "
          "relevant citations, actionable insights, and key findings."),
    backstory=(
        "You are a seasoned research analyst with years of experience in sourcing, verifying, "
        "and synthesizing complex information. Your expertise enables you to deliver precise, "
        "well-structured reports that are both insightful and actionable."
    ),
    llm=llm,
    tools=[EXXAnswerTool()],
    verbose=True
)

# ---------- Agent Task Configuration ---------------------
research_task = Task(
    description=(
        "Research and provide a scholarly article on the given {topic}. "
        "Ensure that the articles are no more than 10 years old. "
        "Do no more than 3 - 5 searches in total (each with a query_number) and then combine the results."
    ),
    expected_output="""A comprehensive executive summary of the research findings.
The summary should be concise and focus on the most important aspects of the research.
The format should be in Markdown (without using "```" tags) and include the following sections:
- Name of the Article 
- All Authors
- Date Published
- Abstract of the Article
- Detailed summary of the Introduction section & source
- Results of the Research
- Key Findings
- Relevant Citations
- Source URLs for all articles used to create your final answer
""",
    agent=research_agent,
    callback=callback_function,
    output_file="research_report.md",
)

# ---------- Crew Configuration ---------------------
research_crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
)

# ---------- Run the Crew ---------------------
input_topic = {"topic": input("Enter a topic to research:\n")}
research_crew.kickoff(inputs=input_topic)

# What are the most major health disparities affecting african american in america
# find relevant research on phytoplankton
