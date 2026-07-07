"""
One-time test script to verify all 4 API keys are working.
This is NOT part of the final project - just a sanity check.
Run it once, confirm everything passes, then you can delete it if you want
(or keep it around, it's harmless).
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads your .env file and makes those values available
# via os.getenv() below - this is why python-dotenv is in requirements.txt
load_dotenv()

print("Checking that all 4 keys are present in .env...\n")

# Step 1: Check each key actually exists (isn't blank or missing)
required_keys = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "LANGCHAIN_API_KEY",
]

all_present = True
for key_name in required_keys:
    value = os.getenv(key_name)
    if value and value.strip() and "your_" not in value and "your-" not in value:
        # Only show first 8 characters for safety - never print full keys
        print(f"  [OK] {key_name} found (starts with {value[:8]}...)")
    else:
        print(f"  [MISSING] {key_name} is missing or still has placeholder text")
        all_present = False

if not all_present:
    print("\nStop here - fix the missing key(s) in your .env file before continuing.")
    exit(1)

print("\nAll 4 keys are present. Now testing each API actually responds...\n")

# Step 2: Actually call each API with a tiny, cheap request to confirm the key works
# (not just that it's typed in - that it's a REAL, active key)

# --- Test OpenAI ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test ok' and nothing else"}],
        max_tokens=10,
    )
    print(f"  [OK] OpenAI responded: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  [FAILED] OpenAI error: {e}")

# --- Test Anthropic ---
# Using claude-haiku-4-5 - the current fast/cheap Claude model as of mid-2026.
# (claude-3-5-haiku-20241022 was retired Feb 19, 2026 - don't use it anymore)
try:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say 'test ok' and nothing else"}],
    )
    print(f"  [OK] Anthropic responded: {response.content[0].text.strip()}")
except Exception as e:
    print(f"  [FAILED] Anthropic error: {e}")

# --- Test Gemini ---
# Using gemini-2.5-flash - Google's current stable, fast, cheap model.
# (gemini-1.5-flash is an older generation - avoid it in new projects)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Say 'test ok' and nothing else")
    print(f"  [OK] Gemini responded: {response.text.strip()}")
except Exception as e:
    print(f"  [FAILED] Gemini error: {e}")

# --- Test LangSmith ---
try:
    from langsmith import Client
    client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
    # list_projects just confirms the key can authenticate - doesn't need any actual projects
    list(client.list_projects(limit=1))
    print(f"  [OK] LangSmith authenticated successfully")
except Exception as e:
    print(f"  [FAILED] LangSmith error: {e}")

print("\nDone. If all 4 show [OK], you're ready to move on to building the agents.")