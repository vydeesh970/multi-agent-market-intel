"""
The Analyst Agent
------------------
This is the second of your 4 agents. Its job is to take the RAW facts the
Researcher gathered and turn them into structured insight - comparisons,
trends, and standout differences between the companies. It does NOT go
search the web itself; it works from what the Researcher already found.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: Define which LLM powers this agent
# ---------------------------------------------------------------------------
# We use CrewAI's own LLM class here, not LangChain's ChatAnthropic wrapper.
# Here's why: CrewAI routes every model call through a library called
# LiteLLM, which acts as a universal translator to whichever AI provider
# you're using. LiteLLM needs the provider explicitly stated as a PREFIX
# on the model name - "anthropic/claude-haiku-4-5", not just
# "claude-haiku-4-5" - or it has no way of knowing which company's API to
# actually call. When we first tried this with LangChain's ChatAnthropic,
# that provider prefix got lost somewhere in the handoff to LiteLLM, which
# caused the "LLM Provider NOT provided" error. Using CrewAI's own LLM
# class directly lets us set this prefix explicitly and reliably.
#
# (For the Researcher agent, this didn't come up, because "gpt-4o-mini" is
# LiteLLM's default assumption when no prefix is given - so it worked by
# coincidence, not because it was actually configured correctly. We'll fix
# that file too, so both agents follow the same reliable pattern.)
#
# We're using claude-haiku-4-5 here (Anthropic's current fast/cheap model)
# because analysis work benefits from careful, well-structured reasoning,
# and Claude models are generally strong at this kind of comparative
# thinking even at their cheaper tier.
analyst_llm = LLM(
    model="anthropic/claude-haiku-4-5",
    temperature=0.4,  # Slightly higher than Researcher's 0.3 - analysis
                       # benefits from a bit more flexibility in how it
                       # frames comparisons, vs. pure fact retrieval which
                       # wants to be as literal as possible.
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# ---------------------------------------------------------------------------
# STEP 2: Define the agent itself
# ---------------------------------------------------------------------------
analyst_agent = Agent(
    role="Senior Market Analyst",

    goal=(
        "Analyze research findings about AI coding assistants (Cursor, "
        "GitHub Copilot, Windsurf, Claude Code) and produce clear, "
        "structured insights - not just a restatement of facts. Identify "
        "meaningful patterns, standout differentiators between products, "
        "pricing trends, and what these findings would mean for a sales "
        "or product team deciding how to position against these "
        "competitors."
    ),

    backstory=(
        "You are a sharp market analyst who has spent years turning messy "
        "research into clear business insight. You never simply repeat "
        "facts back - your value is in the SO WHAT: why does this matter, "
        "what pattern does this fit into, what should a reader do "
        "differently because of this information. You are careful to "
        "clearly separate what is a verified fact (from the research) "
        "versus what is your own interpretation or inference - you never "
        "blur the two together. You write in a direct, confident, "
        "business-appropriate tone, avoiding hedging language unless "
        "genuine uncertainty exists in the underlying research."
    ),

    llm=analyst_llm,
    verbose=True,
    allow_delegation=False,

    # Note: no tools=[] here - the Analyst intentionally has NO search
    # tool. Its job is to work with what the Researcher already found,
    # not go find new information itself. This keeps the pipeline clean:
    # Researcher gathers, Analyst interprets, each agent stays in its lane.
)


# ---------------------------------------------------------------------------
# Standalone test - using the REAL output from your Researcher agent's
# last successful run, so you can see the Analyst reasoning over genuine
# data instead of placeholder text.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from crewai import Task, Crew

    # This is the actual Researcher output from your last run - copied in
    # directly so we can test the Analyst on real data right now. Once we
    # wire everything together with LangGraph later, this handoff will
    # happen automatically instead of being pasted in manually like this.
    researcher_findings = """
GitHub Copilot Current Pricing:
1. Copilot Pro: $10/month
2. Copilot Pro+: $39/month
3. Copilot Business: $19/user/month
4. Enterprise Pricing: Custom pricing based on usage

Recent Feature Updates:
- GitHub Copilot is transitioning to a usage-based billing model. While
  the base plan pricing remains unchanged, users will now receive AI
  credits based on their usage. Each token is priced according to the
  model used, with 1 AI credit equating to $0.01 USD.
- Users with Copilot Pro+ will receive $39 in "monthly AI credits," which
  can be used for code completions and Next Edit Suggestions. If users do
  not utilize the full allowance, they may not carry over unused credits.

Sources:
- GitHub Features: Plans & Pricing (github.com/features/copilot/plans)
- GitHub Blog: Moving to Usage-Based Billing
- GitHub Documentation: Models and Pricing
"""

    test_task = Task(
        description=(
            f"Here are the Researcher's findings on GitHub Copilot:\n\n"
            f"{researcher_findings}\n\n"
            f"Analyze this information. What does GitHub's move to "
            f"usage-based billing suggest about the direction of the AI "
            f"coding assistant market? What should a competing product "
            f"(like Cursor or Windsurf) consider in response? Be specific "
            f"and back your analysis in the facts given above."
        ),
        expected_output=(
            "A structured analysis with clear sections: (1) what the facts "
            "show, (2) what pattern or trend this suggests, (3) what it "
            "means strategically for competitors. Clearly distinguish "
            "facts from your own interpretation."
        ),
        agent=analyst_agent,
    )

    test_crew = Crew(
        agents=[analyst_agent],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RUNNING STANDALONE TEST OF ANALYST AGENT")
    print("=" * 70 + "\n")

    result = test_crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)