"""
The Fact-Checker Agent
------------------------
This is the third of your 4 agents. Its job is genuinely different from
both the Researcher and the Analyst:

  - The Researcher GATHERS information (broad search, discover what's new)
  - The Analyst INTERPRETS information (synthesis, patterns, strategy)
  - The Fact-Checker VERIFIES information (narrow, targeted re-confirmation
    of specific claims, plus a check that the Analyst's conclusions
    actually follow from the facts given)

This agent has its OWN search tool, but uses it differently than the
Researcher does. The Researcher searches broadly to discover what's out
there. The Fact-Checker searches narrowly, to confirm or deny SPECIFIC
claims that have already been made - it's spot-checking, not researching
from scratch.

The Fact-Checker also produces a structured output (a clear CONFIRMED /
FLAGGED verdict per claim) rather than free-flowing prose. This matters
because LangGraph reads this output to make a real decision: if anything
is flagged, loop back to the Researcher for correction (up to a bounded
retry limit); if everything's confirmed, move forward to the Writer.

This version connects to web search via a custom-built MCP (Model Context
Protocol) server, instead of a hand-rolled tool class. The search logic
itself now lives in mcp_servers/search_server.py as a standalone process,
and both this agent and the Researcher connect to that SAME server -
one real source of truth for "how web search works," instead of
duplicated code that could quietly drift apart over time.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM

from mcp_servers.mcp_search_tool import MCPWebSearchTool

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM powers this agent
# ---------------------------------------------------------------------------
# Using Gemini 2.5 Flash here - fast and cheap like our other two agents'
# models, and it completes our 3-provider comparison (OpenAI powers the
# Researcher, Anthropic powers the Analyst, Gemini powers the Fact-Checker).
#
# We use CrewAI's own LLM class (not LangChain's wrapper) because CrewAI
# routes every model call through LiteLLM, which needs the provider
# explicitly stated as a PREFIX on the model name - "gemini/gemini-2.5-flash",
# not just "gemini-2.5-flash" - or it doesn't know which company's API to
# call.
fact_checker_llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.2,  # Lowest temperature of all 4 agents. Fact-checking
                       # is the one job where you want ZERO creative
                       # flexibility - just careful, literal verification.
                       # Any "creativity" here would work against the
                       # whole point of this agent.
    api_key=os.getenv("GOOGLE_API_KEY"),
    # num_retries tells LiteLLM to automatically retry a failed call
    # instead of immediately raising an error. This matters specifically
    # for Gemini's FREE TIER, which allows only 5 requests per MINUTE -
    # the Fact-Checker naturally makes several calls in quick succession
    # (search, reason, search again), and can hit that ceiling even
    # though each individual call is legitimate. LiteLLM automatically
    # waits and retries when it detects a rate-limit (429) error, using
    # the wait time the API itself recommends.
    #
    # NOTE: this does NOT help with Gemini's separate DAILY quota (20
    # requests/day on the free tier) - that one just needs to not be
    # exhausted. If you hit that specific error, the practical fix is
    # waiting ~24 hours for it to reset, or temporarily swapping this
    # agent to a different provider (e.g. model="anthropic/claude-haiku-4-5"
    # with api_key=os.getenv("ANTHROPIC_API_KEY")) to keep testing.
    num_retries=3,
)

# ---------------------------------------------------------------------------
# STEP 2: Connect to the web search MCP server
# ---------------------------------------------------------------------------
# MCPWebSearchTool internally launches mcp_servers/search_server.py as a
# separate process and talks to it over the MCP protocol (stdio
# transport). From this agent's perspective, this is just a normal
# CrewAI tool with a name, a description, and something that returns
# text - it doesn't know or care that the actual search logic lives in a
# completely separate program.
search_tool = MCPWebSearchTool()

# ---------------------------------------------------------------------------
# STEP 3: Define the agent itself
# ---------------------------------------------------------------------------
fact_checker_agent = Agent(
    role="Senior Fact-Checking Editor",

    goal=(
        "Verify the accuracy of specific factual claims made in research "
        "and analysis about AI coding assistants. For each significant "
        "claim (pricing, feature availability, dates, statistics), "
        "confirm it's still accurate by searching for current "
        "confirmation. Also check whether analytical conclusions actually "
        "follow logically from the facts they claim to be based on. "
        "Produce a clear, structured verdict for each claim checked."
    ),

    backstory=(
        "You are a rigorous fact-checking editor with a background in "
        "investigative journalism. Your job is to be professionally "
        "skeptical - you do not simply agree that something looks "
        "correct because it's well-written or confidently stated. You "
        "independently verify specific, checkable claims by searching "
        "for current confirmation rather than trusting the source "
        "material at face value. You are especially alert to: outdated "
        "information presented as current, conclusions that overstate "
        "what the underlying facts actually support, and vague claims "
        "that sound authoritative but aren't actually verifiable. When "
        "you flag something, you explain exactly what's wrong and what "
        "the correct information appears to be instead. You are concise - "
        "you do not repeat back everything that was already correct at "
        "length, you focus your attention on what needs verification or "
        "correction."
    ),

    llm=fact_checker_llm,
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Standalone test - checking the Analyst's actual output from an earlier
# run. This is a good real test because the Analyst made several specific,
# checkable claims (like the $39/month Pro+ price and the 290% increase
# calculation) mixed with genuine interpretation - a good Fact-Checker
# should be able to tell these apart.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from crewai import Task, Crew

    analyst_output = """
GitHub Copilot Current Pricing:
- Copilot Pro: $10/month
- Copilot Pro+: $39/month
- Copilot Business: $19/user/month
- Enterprise: Custom pricing

The analysis states that GitHub Copilot's move from Pro ($10/month) to
Pro+ ($39/month) represents a 290% price increase, and that Copilot Pro+
users receive $39 in monthly AI credits that do not roll over to the
next month if unused.
"""

    test_task = Task(
        description=(
            f"Fact-check the following claims from a market analysis:\n\n"
            f"{analyst_output}\n\n"
            f"Specifically verify: (1) Is the current Copilot Pro+ price "
            f"actually $39/month? (2) Is the 290% price increase "
            f"calculation from $10 to $39 mathematically correct? "
            f"(3) Is it accurate that unused credits do not roll over? "
            f"Search to confirm what you can, and clearly state whether "
            f"each claim is CONFIRMED or FLAGGED, with a brief reason."
        ),
        expected_output=(
            "A structured list of each claim checked, marked CONFIRMED "
            "or FLAGGED, with a brief explanation for each verdict."
        ),
        agent=fact_checker_agent,
    )

    test_crew = Crew(
        agents=[fact_checker_agent],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RUNNING STANDALONE TEST OF FACT-CHECKER AGENT")
    print("=" * 70 + "\n")

    result = test_crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)