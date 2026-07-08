"""
The Researcher Agent
---------------------
This is the first of your 4 agents. Its ONLY job is to gather information -
it does not analyze, fact-check, or write anything.

UPDATED: This version has a real web search tool attached, so it can look
up current information instead of only answering from its training data
(which we saw was stuck at October 2023 in our first test).

NOTE ON THE SEARCH TOOL: we discovered that the pre-built SerperDevTool
from the `crewai_tools` package is currently broken - it crashes on import
because one unrelated tool inside that package (AIMindTool) references a
class (EnvVar) that doesn't exist in our version of crewai. Rather than
fight version compatibility further, we wrote our own small search tool
below, using crewai's own BaseTool class directly. This is actually
BETTER for understanding how tools really work - you can see exactly
what happens instead of trusting a black-box import.
"""

import os
import requests
from dotenv import load_dotenv
from crewai import Agent
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM (AI model) powers this agent
# ---------------------------------------------------------------------------
researcher_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# STEP 2: Build our own web search tool
# ---------------------------------------------------------------------------
# Every CrewAI tool is a Python class that inherits from BaseTool and
# defines 3 things:
#   - name: a short label the agent sees when deciding which tool to use
#   - description: tells the LLM WHEN and WHY to use this tool - this text
#     is genuinely important, the model reads it to decide whether this
#     tool is relevant to what it's currently trying to do
#   - _run(): the actual Python code that executes when the agent calls
#     this tool. Whatever this function returns gets fed back to the agent
#     as text it can read and reason about.
class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Searches the web using Google and returns current, real-time "
        "results. Use this whenever you need up-to-date information such "
        "as current pricing, recent news, or recent feature launches - "
        "do not rely on your training data for anything time-sensitive."
    )

    def _run(self, query: str) -> str:
        """
        This method runs the actual search. 'query' is whatever search
        text the agent decides to send - the agent writes this itself
        based on what it's trying to find out.
        """
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }
        payload = {"q": query}

        response = requests.post(url, headers=headers, json=payload, timeout=15)

        if response.status_code != 200:
            return f"Search failed with status {response.status_code}: {response.text}"

        data = response.json()

        # Serper returns several sections (organic results, knowledge graph,
        # etc.) - we're pulling out the "organic" (regular) search results,
        # which is what we want for research purposes. We take the top 5
        # so we don't overwhelm the agent with too much text at once.
        organic_results = data.get("organic", [])[:5]

        if not organic_results:
            return "No search results found for this query."

        # Format the results into clean, readable text for the agent to read.
        formatted = []
        for i, result in enumerate(organic_results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description available")
            link = result.get("link", "No link")
            formatted.append(f"{i}. {title}\n   {snippet}\n   Source: {link}")

        return "\n\n".join(formatted)


search_tool = WebSearchTool()

# ---------------------------------------------------------------------------
# STEP 3: Define the agent itself
# ---------------------------------------------------------------------------
researcher_agent = Agent(
    role="Senior Market Research Analyst",

    goal=(
        "Gather current, factual, and specific information about AI coding "
        "assistants — Cursor, GitHub Copilot, Windsurf, and Claude Code. "
        "Focus on: pricing changes, new feature launches, user sentiment, "
        "and any notable news from the last few months. ALWAYS search for "
        "current information rather than relying on prior knowledge, since "
        "pricing and features in this market change frequently."
    ),

    backstory=(
        "You are a meticulous market researcher with 10 years of experience "
        "covering the developer tools industry. You have a reputation for "
        "never repeating a rumor as fact, and you never answer from memory "
        "alone when a search tool is available - you always search first "
        "to confirm current information, since AI tools change pricing and "
        "features constantly. You prioritize primary sources (official "
        "pricing pages, official changelogs, official blog posts) over "
        "secondhand summaries or forum speculation. You always mention "
        "which sources your information came from. You organize your "
        "findings clearly by company so the Analyst who reads your work "
        "doesn't have to untangle it."
    ),

    llm=researcher_llm,
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Standalone test - testing something TIME-SENSITIVE on purpose, so we can
# clearly see whether the agent is actually searching for current info vs.
# falling back on stale training knowledge.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from crewai import Task, Crew

    test_task = Task(
        description=(
            "Search for and report on the most recent pricing and any "
            "recent feature updates for GitHub Copilot. Make sure your "
            "information is current - search for it, don't rely on what "
            "you already know. Mention the source of your information."
        ),
        expected_output=(
            "A short, clearly organized summary of GitHub Copilot's "
            "current pricing tiers and any recent feature updates, "
            "with the source noted."
        ),
        agent=researcher_agent,
    )

    test_crew = Crew(
        agents=[researcher_agent],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RUNNING STANDALONE TEST OF RESEARCHER AGENT (WITH WEB SEARCH)")
    print("=" * 70 + "\n")

    result = test_crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)