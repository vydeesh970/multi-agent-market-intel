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
from dotenv import load_dotenv
from crewai import Agent, LLM

from mcp_servers.mcp_search_tool import MCPWebSearchTool

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM (AI model) powers this agent
# ---------------------------------------------------------------------------
# We use CrewAI's own LLM class here. CrewAI routes every model call
# through a library called LiteLLM, which needs the provider explicitly
# stated as a PREFIX on the model name - "openai/gpt-4o-mini", not just
# "gpt-4o-mini" - so it knows which company's API to call. OpenAI happens
# to be LiteLLM's silent default, so leaving off the prefix worked here by
# coincidence - but being explicit is the correct, reliable pattern, and
# it's REQUIRED for other providers like Anthropic (as you'll see in the
# Analyst agent), so we're using it consistently across every agent.
researcher_llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.3,  # Lower temperature = more focused, factual, less "creative"
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# STEP 2: Connect to the web search MCP server
# ---------------------------------------------------------------------------
# This used to be a hand-rolled WebSearchTool class defined right here in
# this file, calling the Serper API directly. It's now been upgraded to
# MCPWebSearchTool, which internally launches mcp_servers/search_server.py
# as a separate process and talks to it over the MCP protocol. From this
# agent's perspective, NOTHING else changes - it's still just a tool with
# a name, a description, and something that returns text. That's the
# whole point of MCP: the agent doesn't need to know or care that the
# actual search logic now lives in a completely separate program.
search_tool = MCPWebSearchTool()

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