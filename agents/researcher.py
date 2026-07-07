"""
The Researcher Agent
---------------------
This is the first of your 4 agents. Its ONLY job is to gather information -
it does not analyze, fact-check, or write anything. Keeping agents narrowly
focused like this (instead of one agent doing everything) is the whole
point of a multi-agent system: each agent gets really good at one job,
the same way a real company has separate researchers, analysts, and writers
instead of one person doing all three badly.
"""

import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

# load_dotenv() reads your .env file so os.getenv() below can find your API keys.
# We call this at the top of every file that needs API keys.
load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM (AI model) powers this agent
# ---------------------------------------------------------------------------
# ChatOpenAI is LangChain's "wrapper" around OpenAI's API - it translates
# CrewAI's internal calls into the format OpenAI's API expects.
#
# We're using gpt-4o-mini here because:
#   - It's OpenAI's cheap, fast model - good for research/gathering tasks
#     that don't need deep reasoning, just accurate retrieval and summarizing.
#   - Later, you'll swap this out for claude-haiku-4-5 or gemini-2.5-flash
#     on the SAME agent to compare cost/quality - that's your head-to-head
#     model comparison for the portfolio story.
researcher_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,  # Lower temperature = more focused, factual, less "creative"
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# STEP 2: Define the agent itself
# ---------------------------------------------------------------------------
researcher_agent = Agent(
    role="Senior Market Research Analyst",

    goal=(
        "Gather current, factual, and specific information about AI coding "
        "assistants — Cursor, GitHub Copilot, Windsurf, and Claude Code. "
        "Focus on: pricing changes, new feature launches, user sentiment, "
        "and any notable news from the last few months."
    ),

    backstory=(
        "You are a meticulous market researcher with 10 years of experience "
        "covering the developer tools industry. You have a reputation for "
        "never repeating a rumor as fact. You always try to identify WHEN "
        "a piece of information was true - since AI tools change pricing "
        "and features constantly, a fact from 6 months ago may already be "
        "outdated. You prioritize primary sources (official pricing pages, "
        "official changelogs, official blog posts) over secondhand summaries "
        "or forum speculation. You organize your findings clearly by company "
        "so the Analyst who reads your work doesn't have to untangle it."
    ),

    llm=researcher_llm,

    # verbose=True makes CrewAI print out the agent's step-by-step thinking
    # process to your terminal as it runs. This is incredibly useful while
    # building and debugging - you can literally watch the agent "think."
    # We'll likely turn this off (set to False) once everything is stable,
    # since it produces a lot of terminal output.
    verbose=True,

    # allow_delegation=False means this agent cannot hand off work to other
    # agents on its own. We want strict control over the flow (Researcher
    # then Analyst then Fact-Checker then Writer) via LangGraph later, not
    # agents freelancing and delegating to each other unpredictably.
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Quick standalone test - lets us confirm this ONE agent works before we
# build the other 3 and wire everything together with LangGraph.
# This block only runs if you execute this file directly
# (python agents/researcher.py) - it will NOT run when other files import
# researcher_agent from this file later.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from crewai import Task, Crew

    # A Task is a specific assignment given to an agent - separate from the
    # agent's general role/goal/backstory. Think of the agent as "a person
    # with a job title" and the Task as "today's specific to-do item."
    test_task = Task(
        description=(
            "Research the current pricing model for GitHub Copilot. "
            "List the different pricing tiers and what each includes."
        ),
        expected_output=(
            "A short, clearly organized list of GitHub Copilot's pricing "
            "tiers and what's included in each."
        ),
        agent=researcher_agent,
    )

    # A Crew is a group of agents working on a set of tasks. Right now we
    # only have 1 agent and 1 task - this is just a smoke test to confirm
    # the agent runs end-to-end before we build the full 4-agent crew.
    test_crew = Crew(
        agents=[researcher_agent],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RUNNING STANDALONE TEST OF RESEARCHER AGENT")
    print("=" * 70 + "\n")

    result = test_crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)