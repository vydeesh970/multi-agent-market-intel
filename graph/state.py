"""
The State Definition
----------------------
This defines the exact SHAPE of the shared "notebook" that flows through
every node in the graph. Every node reads some fields from this and writes
some fields back to it. Defining this shape explicitly with TypedDict
(instead of a loose, undefined dictionary) means Python and your editor
can catch a typo'd key name before you ever run the code - a huge
practical benefit once you have 4+ nodes all touching the same object.
"""

from typing import TypedDict


class MarketIntelState(TypedDict):
    # ---- INPUT: what we're researching ----
    company: str  # e.g. "GitHub Copilot" - set once at the very start

    # ---- FILLED IN BY EACH NODE AS THE PIPELINE RUNS ----
    research_findings: str      # Researcher node writes this
    analysis: str                # Analyst node writes this
    fact_check_verdict: str      # Fact-Checker node writes this (the full text)
    fact_check_passed: bool      # Fact-Checker node writes this (True/False)
    final_report: str            # Writer node writes this - the end product

    # ---- LOOP SAFEGUARD ----
    # This is the bounded-retry mechanism we discussed: if the Fact-Checker
    # keeps flagging problems, we don't want the graph to loop back to the
    # Researcher forever. retry_count tracks how many times we've looped
    # back so far; max_retries is the hard ceiling. Once retry_count hits
    # max_retries, we force the graph to proceed to the Writer anyway
    # (with whatever caveats the Fact-Checker last reported), rather than
    # spinning indefinitely and burning API credits.
    retry_count: int
    max_retries: int