"""
The Graph Nodes
-----------------
Each function here wraps one of your 4 CrewAI agents into a shape
LangGraph can call as a single "step." Every node function follows the
exact same pattern:

  1. Receive the current State (a dict-like object)
  2. Pull whatever input it needs out of State
  3. Build a CrewAI Task using that input
  4. Run a single-agent Crew to get a result
  5. Write the result back into State, and return the updated State

LangGraph calls these functions for you, in the order defined by the
graph's edges (which we'll build in build_graph.py) - you never call these
functions directly yourself.
"""

from crewai import Task, Crew

from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.fact_checker import fact_checker_agent
from agents.writer import writer_agent

from graph.state import MarketIntelState


def research_node(state: MarketIntelState) -> MarketIntelState:
    """
    Runs the Researcher agent. Pulls the company name from State, searches
    for current info, and writes the findings back into State under
    'research_findings' for the next node (Analyst) to use.
    """
    print("\n>>> ENTERING NODE: Researcher\n")

    task = Task(
        description=(
            f"Search for and report on the most recent pricing and any "
            f"recent feature updates for {state['company']}. Make sure "
            f"your information is current - search for it, don't rely on "
            f"what you already know. Mention the source of your "
            f"information."
        ),
        expected_output=(
            f"A short, clearly organized summary of {state['company']}'s "
            f"current pricing tiers and any recent feature updates, with "
            f"sources noted."
        ),
        agent=researcher_agent,
    )

    crew = Crew(agents=[researcher_agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    # CrewOutput objects (what .kickoff() returns) can be converted to a
    # plain string with str() - we do this so State only ever holds simple
    # strings, not CrewAI-specific objects, keeping State easy to inspect
    # and print at any point.
    state["research_findings"] = str(result)
    return state


def analyze_node(state: MarketIntelState) -> MarketIntelState:
    """
    Runs the Analyst agent. Pulls the Researcher's findings from State
    (no manual copy-pasting required anymore - this is the automatic
    hand-off we built LangGraph specifically to achieve), and writes the
    analysis back into State under 'analysis'.
    """
    print("\n>>> ENTERING NODE: Analyst\n")

    task = Task(
        description=(
            f"Here are the Researcher's findings on {state['company']}:\n\n"
            f"{state['research_findings']}\n\n"
            f"Analyze this information. What patterns or trends does this "
            f"suggest about the AI coding assistant market? What should "
            f"competing products consider in response? Be specific and "
            f"back your analysis in the facts given above."
        ),
        expected_output=(
            "A structured analysis clearly separating verified facts from "
            "your own interpretation, with strategic implications for "
            "competitors."
        ),
        agent=analyst_agent,
    )

    crew = Crew(agents=[analyst_agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    state["analysis"] = str(result)
    return state


def fact_check_node(state: MarketIntelState) -> MarketIntelState:
    """
    Runs the Fact-Checker agent against the Analyst's output. This node
    does something the other nodes don't: it also decides whether the
    pipeline PASSED or needs to loop back, by writing a boolean into
    State ('fact_check_passed'). This is what the conditional edge (built
    in build_graph.py) will read to decide where to send the pipeline
    next.

    NOTE ON THE PASS/FAIL CHECK: right now we use a simple, honest
    heuristic - checking whether the word "FLAGGED" appears anywhere in
    the Fact-Checker's response. This works because we explicitly told
    the Fact-Checker agent (in its Task instructions) to use that exact
    word for anything it couldn't confirm. This is a reasonable starting
    point, but it's worth knowing its limitation: it's a text-matching
    check, not true structured data. A more robust version (something to
    consider as a future improvement) would have the Fact-Checker return
    actual structured JSON with a real boolean field, instead of us
    inferring pass/fail from free-form text.

    This node ALSO handles incrementing retry_count when a failure is
    detected - keeping this state mutation inside a node (not inside the
    conditional routing function) follows LangGraph's intended pattern.
    """
    print("\n>>> ENTERING NODE: Fact-Checker\n")

    task = Task(
        description=(
            f"Fact-check the following analysis about {state['company']}:\n\n"
            f"{state['analysis']}\n\n"
            f"Verify any specific, checkable claims (prices, statistics, "
            f"dates) by searching for current confirmation. Clearly state "
            f"whether each claim is CONFIRMED or FLAGGED, with a brief "
            f"reason. Use the word FLAGGED explicitly for anything you "
            f"cannot confirm."
        ),
        expected_output=(
            "A structured list of claims checked, each marked CONFIRMED "
            "or FLAGGED with a brief explanation."
        ),
        agent=fact_checker_agent,
    )

    crew = Crew(agents=[fact_checker_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    result_text = str(result)

    state["fact_check_verdict"] = result_text
    state["fact_check_passed"] = "FLAGGED" not in result_text.upper()

    if not state["fact_check_passed"]:
        state["retry_count"] += 1
        print(
            f"\n>>> Fact-check found issues. Retry count is now "
            f"{state['retry_count']} of {state['max_retries']} allowed.\n"
        )

    return state


def write_node(state: MarketIntelState) -> MarketIntelState:
    """
    Runs the Writer agent - the final step. Pulls research, analysis, AND
    the fact-check verdict from State, so the Writer can apply the
    confidence-labeling editorial policy we designed (confirmed facts
    stated plainly, flagged claims included with caveats, nothing
    silently dropped).
    """
    print("\n>>> ENTERING NODE: Writer\n")

    task = Task(
        description=(
            f"Write a market intelligence report section about "
            f"{state['company']}, using the following inputs from your "
            f"team:\n\n"
            f"RESEARCH FINDINGS:\n{state['research_findings']}\n\n"
            f"STRATEGIC ANALYSIS:\n{state['analysis']}\n\n"
            f"FACT-CHECK VERDICTS:\n{state['fact_check_verdict']}\n\n"
            f"Write this as a professional report section. Any claim "
            f"marked FLAGGED by the fact-checker must be included with a "
            f"clear confidence caveat, never stated as a plain fact."
        ),
        expected_output=(
            "A well-formatted market intelligence report section, with "
            "confident language for confirmed facts, appropriate caveats "
            "for flagged claims, and a sources note."
        ),
        agent=writer_agent,
    )

    crew = Crew(agents=[writer_agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    state["final_report"] = str(result)
    return state