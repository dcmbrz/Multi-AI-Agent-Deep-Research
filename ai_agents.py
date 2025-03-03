import streamlit as st
st.set_page_config(page_title= "Deep Researching Agent mk1", page_icon=":mortar_board", layout="wide")


from crewai import Agent, Task, Crew, LLM, Process
from crewai.project import CrewBase, agent, crew
from crewai.tasks.task_output import TaskOutput
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool, WebsiteSearchTool
from exa_py import Exa
from typing import Type, Union
import requests
import logging
from scripts.regsetup import description

# ---------- Streamlit UI ---------------------
st.title("Deep Researching Agent mk1")

# sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Choose model",
    ["OpenAI 4o mini", "Local DeepSeek r-1"]

)



load_dotenv()
# --- Environment Variable Checks (New) ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
if not os.getenv("EXA_API_KEY"):
    raise ValueError("Missing EXA_API_KEY environment variable.")


# ---------- Enhanced search tool ---------------------
search_tool = SerperDevTool()
website_tool = WebsiteSearchTool()


# Setting up a logger for debugging and production logging.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- LLM Configuration ---------------------
def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_llm(gpt=True):
    if gpt:
        return ChatOpenAI(model="gpt-4o-mini")

    llm = LLM(
        model="ollama/deepseek-r1:latest",
        base_url="http://localhost:11433",
        temperature=0.7
    )
    return llm


# ---------- EXA Answer Tool ---------------------
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

        if get_llm() and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key was not found")

        if not get_llm() and not check_ollama_availability():
            raise ValueError("Ollama server not running")


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
    with st.status("Generating Research Report...", expanded=True) as status:
        st.markdown(f"""
                Agent: {output.agent}
                Task: {output.description}
                Task Summary: {output.summary}
                Output: {output.raw}
                """)

# ---------- Research Agents Configuration ---------------------
research_agent = Agent(
    role="Deep Research Specialist",
    goal="Conduct a comprehensive research and gather detailed information",
    backstory="""Expert researcher skilled at discovering hard-to-find information
    and connecting complex data points on the {topic}. Specializes in thorough, detailed research 
    on the {topic}.""",
    tools=[search_tool, website_tool],
    llm=get_llm(),
    verbose=True,
    max_iter=15,
    allow_delegation=False
)

analyst_agent = Agent(
    role="Research Analyst",
    goal=("Conduct thorough research on the given {topic}, providing detailed analysis, "
          "relevant citations, actionable insights, and key findings."
          "You will also analyze and synthesize research findings"),
    backstory=(
        "You are a seasoned research analyst with years of experience in sourcing, verifying, "
        "and synthesizing complex information. Your expertise enables you to deliver precise, "
        "well-structured reports that are both insightful and actionable."
        "You are also skilled at identifying key patterns and insights. You specialize in clear and actionable analysis."
    ),
    llm=get_llm(),
    tools=[search_tool],
    verbose=True,
    max_iter=10,
    allow_delegation=False
)

writing_agent = Agent(
    role='Content Synthesizer',
    goal="Create clear, structured reports from Research Analyst",
    backstory="""Expert writer skilled at transforming complex analysis into clear, 
    engaging content while maintaining technical accuracy""",
    llm=get_llm(),
    verbose=True,
    max_iter=8,
    allow_delegation=False
)




# ---------- Agent Task Configuration ---------------------
research_task = Task(
    description= """
    The primary role of the researcher agent is to gather and compile accurate, reliable data from relevant research 
    articles. Ensure that the articles are no more than 10 years old. Your focus should be on collecting key information
    without delving into analysis or narrative composition.
    Please concentrate on the following components:

        1. Article Title(s):
            - Extract the title(s) of each research article.
        
        2. Authors:
            - List the full names of all authors involved.
        
        3. Publication Date:
            - Record the date each article was published.
        
        4. Abstract:
            - Retrieve the original abstract provided in the article.
        
        5. Introduction Overview:
            - Summarize the main points of the introduction section, and include the source of this summary.
        
        6. Research Results:
            - Extract and list the main outcomes and data points reported.
        
        7. Key Findings:
            - Identify the critical discoveries and conclusions drawn from the research.
        
        8. Relevant Citations:
            - Collect citations that support the core information extracted from the articles.
        
        9. Source URLs:
            - Provide the direct URLs for all articles used.
            - Please present all collected data in a structured Markdown format (without using "```" tags). Focus solely 
            on data collection to ensure the subsequent analyzer and writing agents have a comprehensive and reliable 
            foundation to build upon.""",
    agent=research_agent,
    expected_output="Detailed research findings following give instructions",
    callback=callback_function
)


analysis_task = Task(
    description= """ 
    The analysis agent plays a crucial role in bridging the gap between raw research data and the 
    final narrative. Your output will serve as the foundation for the writing agent, so clarity and structure are 
    paramount. Your responsibilities include:

    1. Data Synthesis:
        - Review all information provided by the researcher agent (article titles, authors, publication dates, abstracts,
        introduction overviews, research results, key findings, citations, and source URLs).
        - Identify common themes, patterns, and any inconsistencies across multiple sources.
    
    2. Critical Evaluation:
        - Assess the credibility, relevance, and overall quality of the collected research data.
        - Evaluate the significance of the findings in relation to the research question, noting strengths, 
        weaknesses, and any gaps.
    
    3. Insight Generation:
        - Distill your analysis into clear, concise insights and key takeaways.
        - Connect the dots between disparate pieces of data to create a coherent picture of the research landscape.
        - Clearly mark areas that may benefit from further exploration.
        
    4. Preparation for the Writing Agent:
        - Format your analysis in structured Markdown, using headings and bullet points to organize the content.
        - Ensure that your output is comprehensive yet straightforward, providing the writing agent with all necessary 
        context for crafting the final narrative without additional interpretation.""",
    agent= analyst_agent,
    context=[research_task],
    expected_output="Analysis of research findings and insights",
    callback=callback_function
)

writing_task = Task(
    description=(
        """
        Your role is to transform the structured analysis provided by the analysis agent into a polished, coherent 
        narrative. Use the analysis agent’s output as your sole source of information, ensuring that you maintain 
        accuracy while presenting the findings in an engaging manner. Follow these guidelines:

        Review the Analysis Agent’s Output:
            - Thoroughly read the research analysis summary, which includes the overview, common themes, critical 
            evaluations, key insights, and any recommendations or identified gaps.
            - Use this information as the foundation for your narrative, ensuring that no new data or interpretations are 
            introduced.

        Structure and Formatting:
            - Format your narrative in Markdown, using clear headings, subheadings, and bullet points where appropriate.
            - Start with a compelling introduction that sets the context for the research.
            - Develop a well-organized body that presents the synthesized data, critical evaluations, and insights.
            - Conclude with a summary that reinforces the key findings and outlines any next steps or areas for further 
              investigation.

        Clarity and Engagement:
            - Write in a clear, concise, and engaging style that is accessible to a broad audience.
            - Ensure that the narrative flows logically from the introduction to the conclusion.
            - Emphasize the most significant insights and key takeaways derived from the analysis.

        Consistency with Source Data:
            - Maintain fidelity to the analysis agent’s content; your narrative should be a re-articulation of the 
                provided analysis, not an expansion with new ideas.
            - Verify that all the details, such as themes, critical evaluations, and citations, are accurately represented.


        By following these guidelines, you will produce a final narrative that effectively communicates the research 
        findings and insights in a clear, engaging, and well-structured format.
"""
    ),

    expected_output="""
    A polished, comprehensive executive summary of the research findings derived from the research and analysis 
    agent's output. The summary should be concise, engaging, and focus on the most important aspects of the research.
    The format must be in Markdown (without using "```" tags) and include the following sections:

        Introduction:
            - Provide a compelling context for the research, setting the stage for the insights that follow.

        Research Overview:
            - Article Titles: List the names of the articles.
            - Authors: Include all the authors involved.
           -  Publication Dates: State the publication date(s) for the articles.
            - Abstracts: Present the abstracts as provided in the original articles.

        Detailed Analysis:
            - Introduction Summaries & Sources: Offer a concise summary of the introduction sections along with
                source references.
            - Research Results: Detail the key outcomes and data points from the studies.
            - Key Findings: Highlight the primary discoveries and conclusions drawn from the research.

        Critical Evaluation & Insights:
            - Summarize common themes, patterns, and any inconsistencies noted in the data.
            - Provide clear, actionable insights and recommendations for future exploration based on the analysis.

        Citations:
            - Include all relevant citations that support the analysis.

        Source URLs:
            - List the direct URLs for all articles and sources used to compile the final narrative.

    The narrative should follow a logical structure: begin with an engaging introduction, transition into a 
    well-organized body that presents the synthesized data and critical evaluations, and conclude with a summary that
    reinforces the key takeaways. Ensure that the content is accurate, cohesive, and accessible to a broad audience.""",
    agent=writing_agent,
    context=[research_task, analysis_task],
    callback=callback_function,
    output_file="research_report.md",
)

# ---------- Crew Configuration ---------------------
research_crew = Crew(
    agents=[research_agent, analyst_agent, writing_agent],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

# ---------- Run the Crew ---------------------
input_topic = {"topic": st.text_input("Enter a topic to research:\n")}
if st.button("Begin Research") and input_topic:
    with st.spinner("Research under way..."):
        research_crew.kickoff(inputs=input_topic)



