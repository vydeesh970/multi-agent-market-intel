"""
Running the Full Pipeline
----------------------------
This is the script you actually run. It builds the compiled graph, sets
up the STARTING state (just the company name and the retry counters -
everything else gets filled in automatically as the graph runs), and
kicks it off.

Compare this to how you were testing each agent before: previously, you
ran 4 separate scripts and manually copied each agent's output into the
next one's test input. Now, one single .invoke() call runs all 4 agents
in the correct order, with real hand-offs happening automatically through
the shared State object. This is the actual payoff of everything we've
built today.
"""

from dotenv import load_dotenv
load_dotenv()

from graph.build_graph import build_market_intel_graph

if __name__ == "__main__":
    app = build_market_intel_graph()

    # This is the STARTING state - notice research_findings, analysis,
    # etc. are all empty strings right now. They get filled in one by one
    # as the graph executes each node in order.
    initial_state = {
        "company": "GitHub Copilot",
        "research_findings": "",
        "analysis": "",
        "fact_check_verdict": "",
        "fact_check_passed": False,
        "final_report": "",
        "retry_count": 0,
        "max_retries": 2,  # Allow up to 2 loop-backs before forcing
                            # the pipeline to proceed anyway.
    }

    print("\n" + "=" * 70)
    print("STARTING FULL MARKET INTELLIGENCE PIPELINE")
    print(f"Target: {initial_state['company']}")
    print("=" * 70)

    # .invoke() runs the ENTIRE graph from START to END, following
    # whatever path the edges (including the conditional one) determine
    # at runtime. It returns the FINAL state after everything has run.
    final_state = app.invoke(initial_state)

    print("\n\n" + "=" * 70)
    print("PIPELINE COMPLETE - FINAL REPORT")
    print("=" * 70 + "\n")
    print(final_state["final_report"])

    print("\n\n" + "=" * 70)
    print("PIPELINE METADATA")
    print("=" * 70)
    print(f"Fact-check passed: {final_state['fact_check_passed']}")
    print(f"Retry attempts used: {final_state['retry_count']}")