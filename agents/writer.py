"""
The Writer Agent
------------------
This is the fourth and final agent in your crew. Its job is to take
everything the other three agents produced - raw research, strategic
analysis, and fact-check verdicts - and weave them into one polished,
coherent market intelligence report that a sales or product person would
actually want to read.

EDITORIAL POLICY (a real design decision, not just a technical detail):
This Writer does NOT silently drop claims that the Fact-Checker flagged
as uncertain. Instead, it includes them with clear confidence labeling -
e.g. "This figure could not be independently confirmed." This mirrors how
real professional intelligence reports work: they communicate confidence
levels rather than pretending everything is equally certain. Silently
dropping uncertain information might look cleaner, but it erodes trust
the first time someone discovers something was missing with no
explanation.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM powers this agent
# ---------------------------------------------------------------------------
# We're intentionally using a STRONGER (and more expensive) model here than
# the other 3 agents. This is a deliberate quality-over-cost tradeoff worth
# understanding: the Researcher, Analyst, and Fact-Checker all do internal
# reasoning work - their output gets consumed by other agents, not directly
# by a human. The Writer's output IS the final product a human reads. This
# is the one place in the pipeline where writing quality (tone, flow,
# clarity) has direct, visible impact on the end user, so it's worth
# spending more here. Cheap models everywhere would save money but produce
# a worse final report; expensive models everywhere would waste money on
# tasks that don't need that extra quality. Matching model cost to task
# importance like this is a real, defensible architecture decision.
writer_llm = LLM(
    model="openai/gpt-4o",
    temperature=0.6,  # Noticeably higher than our other agents. Writing
                       # quality benefits from more natural language
                       # variation - a report that reads like it was
                       # written by a rigid, repetitive process is worse
                       # than one with natural, varied sentence structure.
                       # We can afford more "creativity" here because the
                       # facts have already been gathered and verified by
                       # earlier agents - the Writer's temperature doesn't
                       # affect factual accuracy the way it would if this
                       # were a research agent.
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# STEP 2: Define the agent itself
# ---------------------------------------------------------------------------
writer_agent = Agent(
    role="Senior Market Intelligence Report Writer",

    goal=(
        "Synthesize research findings, strategic analysis, and fact-check "
        "verdicts into a single, polished market intelligence report. "
        "The report should be genuinely useful and readable for a sales "
        "or product team - clear structure, no unnecessary jargon, and "
        "confident where the underlying facts are confirmed. Any claim "
        "that the Fact-Checker flagged as unconfirmed must still be "
        "included, but clearly labeled with a confidence note - never "
        "silently omit uncertain information, and never present "
        "unconfirmed information as if it were certain."
    ),

    backstory=(
        "You are an experienced market intelligence writer who has "
        "produced competitive analysis reports for product and sales "
        "teams for over a decade. You know that a report loses all value "
        "if the reader can't tell what's solid and what's uncertain, so "
        "you are disciplined about confidence labeling - you use phrases "
        "like 'independently confirmed' for verified facts and "
        "'could not be independently confirmed as of this report' for "
        "anything flagged. You write in clear, direct, professional "
        "language - no filler, no unnecessary hedging on things that "
        "ARE confirmed, and no false confidence on things that aren't. "
        "You organize reports so a busy reader can skim the headlines and "
        "structure and still get the key takeaways, while a more careful "
        "reader can dig into the supporting detail. You always include a "
        "clear 'Sources' section and a note of when the report was "
        "generated, since competitive intelligence has a shelf life."
    ),

    llm=writer_llm,
    verbose=True,
    allow_delegation=False,

    # No tools here either, same reasoning as the Analyst - the Writer's
    # job is to synthesize what the other 3 agents already produced, not
    # to go find new information itself.
)


# ---------------------------------------------------------------------------
# Standalone test - feeding it condensed versions of the real outputs from
# your Researcher, Analyst, and Fact-Checker runs, including one claim
# marked as needing a confidence caveat, so we can see whether the Writer
# actually handles that instruction correctly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from crewai import Task, Crew

    researcher_findings = """
GitHub Copilot Current Pricing:
- Copilot Pro: $10/month
- Copilot Pro+: $39/month
- Copilot Business: $19/user/month
- Enterprise: Custom pricing
GitHub is transitioning to usage-based billing. Base plan pricing is
unchanged, but users now receive monthly AI credits tied to their
subscription tier (1 credit = $0.01 USD).
Sources: github.com/features/copilot/plans, github.blog
"""

    analyst_insights = """
GitHub's shift to usage-based billing signals the AI coding assistant
market is maturing around consumption-based monetization rather than
flat subscriptions. Competitors should NOT simply copy this hybrid model -
instead they should pick a clear position: the "simple, predictable"
subscription alternative, or the "transparent, pay-for-what-you-use"
alternative. Matching GitHub's complexity without a clear reason signals
following, not leading.
"""

    fact_check_verdicts = """
- Copilot Pro+ price of $39/month: CONFIRMED via official GitHub docs.
- 290% price increase calculation ($10 to $39): CONFIRMED, math verified.
- Claim that unused AI credits do not roll over month to month: 
  COULD NOT BE INDEPENDENTLY CONFIRMED from an official primary source -
  evidence was indirect (community discussion, inferred from "monthly
  allowance" language), not an explicit official policy statement.
"""

    test_task = Task(
        description=(
            f"Write a market intelligence report section about GitHub "
            f"Copilot, using the following inputs from your team:\n\n"
            f"RESEARCH FINDINGS:\n{researcher_findings}\n\n"
            f"STRATEGIC ANALYSIS:\n{analyst_insights}\n\n"
            f"FACT-CHECK VERDICTS:\n{fact_check_verdicts}\n\n"
            f"Write this as a professional report section. Remember: the "
            f"credit rollover claim was NOT independently confirmed - "
            f"include it, but with a clear confidence caveat, not as a "
            f"stated fact."
        ),
        expected_output=(
            "A well-formatted market intelligence report section on "
            "GitHub Copilot, with clear structure, confident language "
            "for confirmed facts, appropriate caveats for unconfirmed "
            "claims, and a sources note."
        ),
        agent=writer_agent,
    )

    test_crew = Crew(
        agents=[writer_agent],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RUNNING STANDALONE TEST OF WRITER AGENT")
    print("=" * 70 + "\n")

    result = test_crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)