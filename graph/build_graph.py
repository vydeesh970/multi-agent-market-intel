"""
Building the Graph
---------------------
This file wires your 4 nodes together into an actual flowchart, matching
the diagram we discussed:

    START -> Researcher -> Analyst -> Fact-Checker -> [decision point]
                                                              |
                                        flagged claims exist  |  all confirmed
                                        AND retries remain    |  OR retries exhausted
                                                v              v
                                         back to Researcher   Writer -> END
"""

from langgraph.graph import StateGraph, START, END

from graph.state import MarketIntelState
from graph.nodes import research_node, analyze_node, fact_check_node, write_node


def route_after_fact_check(state: MarketIntelState) -> str:
    """
    This is the CONDITIONAL EDGE function - the actual decision point in
    the flowchart. LangGraph calls this after the fact_check_node runs,
    and whatever STRING this function returns tells LangGraph which node
    to go to next. This is the mechanism that makes real branching
    possible, unlike CrewAI's fixed sequential Crew.

    The logic is simple and readable on purpose:
      - If the fact-check passed -> go straight to the Writer
      - If it failed AND we haven't hit max_retries yet -> loop back to
        the Researcher for another attempt
      - If it failed but we've ALREADY hit max_retries -> give up looping
        and go to the Writer anyway. The Writer will still see the
        flagged claims in fact_check_verdict and apply the confidence-
        labeling policy we built - so the pipeline never gets stuck, it
        just proceeds with appropriate caveats instead of perfect
        confirmation.
    """
    if state["fact_check_passed"]:
        print("\n>>> DECISION: Fact-check passed. Routing to Writer.\n")
        return "writer"

    if state["retry_count"] < state["max_retries"]:
        print(
            f"\n>>> DECISION: Fact-check failed (attempt "
            f"{state['retry_count']}/{state['max_retries']}). "
            f"Routing back to Researcher.\n"
        )
        return "researcher"

    print(
        "\n>>> DECISION: Fact-check failed but max retries reached. "
        "Proceeding to Writer with caveats instead of looping further.\n"
    )
    return "writer"


def build_market_intel_graph():
    """
    Assembles the full graph and returns a COMPILED, runnable version of
    it. "Compiling" here means LangGraph validates the structure (checks
    every node is reachable, every edge points somewhere real) and
    prepares it to actually execute - similar in spirit to how you
    wouldn't run raw source code without it being parsed/checked first.
    """

    # StateGraph is the builder object. We tell it what shape of State
    # it'll be passing around, using the MarketIntelState TypedDict we
    # defined earlier.
    graph = StateGraph(MarketIntelState)

    # ---- Register each node under a short string name ----
    # These names are what we'll reference when defining edges below, and
    # they're also what route_after_fact_check() returns to pick the next
    # step. Think of these as labels for each box in the flowchart.
    graph.add_node("researcher", research_node)
    graph.add_node("analyst", analyze_node)
    graph.add_node("fact_checker", fact_check_node)
    graph.add_node("writer", write_node)

    # ---- Define the FIXED edges (no decision involved) ----
    # START is a special LangGraph marker meaning "the graph begins here."
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "fact_checker")

    # ---- Define the CONDITIONAL edge ----
    # After "fact_checker" runs, instead of always going to a fixed next
    # node, LangGraph calls route_after_fact_check(state) and sends the
    # pipeline to whichever node NAME that function returns. The
    # dictionary below maps each possible returned string to the actual
    # node - "researcher" -> loop back, "writer" -> move forward.
    graph.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    # ---- The Writer is the final step ----
    # END is a special LangGraph marker meaning "the graph is done here."
    graph.add_edge("writer", END)

    # Compile turns this definition into something we can actually run.
    return graph.compile()