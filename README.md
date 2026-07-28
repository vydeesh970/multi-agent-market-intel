# Multi-Agent Market Intelligence System

An autonomous, self-correcting multi-agent system that researches, analyzes, fact-checks, and reports on the AI coding assistant market (Cursor, GitHub Copilot, Windsurf, Claude Code) — with zero human intervention after it's kicked off.

Built with **CrewAI**, **LangGraph**, **MCP (Model Context Protocol)**, and orchestrated across **OpenAI, Anthropic, and Google Gemini**, containerized with **Docker**, and deployed as a scheduled **Kubernetes CronJob**.

---

## What it does

Sales and product teams need continuous competitor monitoring, but manual research doesn't scale. This system runs a crew of 4 specialized AI agents that autonomously:

1. **Research** — search the live web for current pricing, features, and news
2. **Analyze** — synthesize raw findings into strategic insight and competitive positioning
3. **Fact-check** — independently re-verify specific claims against live sources, and flag anything it can't confirm
4. **Write** — produce a polished report that clearly separates confirmed facts from flagged, uncertain claims

If the fact-checker finds a problem, the system **automatically loops back** and re-researches — up to a bounded retry limit — before falling back to a transparent, caveated report rather than either failing silently or looping forever.

---

## Architecture

```
                    ┌─────────────┐
    START ─────────▶│  Researcher │──▶ searches the web via a custom MCP server
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Analyst   │──▶ synthesizes findings into strategic insight
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │Fact-Checker │──▶ independently re-verifies specific claims
                    └─────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Conditional Edge │
                  │  (LangGraph)     │
                  └──────────────────┘
                     │            │
        claims flagged │            │ all confirmed
        AND retries left│            │ OR retries exhausted
                     ▼            ▼
              back to Researcher  ┌─────────┐
                                  │ Writer  │──▶ confidence-labeled final report
                                  └─────────┘
                                       │
                                       ▼
                                      END
```

Each agent runs on a different LLM provider — a deliberate choice to compare cost, latency, and quality across providers for different cognitive tasks (see "Key Engineering Decisions" below).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | CrewAI |
| Orchestration / state machine | LangGraph |
| Tool protocol | MCP (Model Context Protocol) — custom-built server + client |
| LLM providers | OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude Haiku 4.5), Google (Gemini 2.5 Flash) |
| Observability | LangSmith |
| Search | Serper API (via custom MCP server) |
| Containerization | Docker |
| Orchestration/deployment | Kubernetes (CronJob), tested via Minikube |
| Language | Python 3.11 |

---

## Key Engineering Decisions

**Multi-provider LLM routing, matched to task type.** The Researcher (GPT-4o-mini) prioritizes cheap, fast retrieval. The Analyst (Claude Haiku 4.5) handles synthesis and reasoning. The Fact-Checker (Gemini 2.5 Flash) does narrow, skeptical re-verification. The Writer (GPT-4o) is the one place a stronger, more expensive model is worth the cost, since it's the only agent whose output a human actually reads directly.

**A confidence-labeling editorial policy, not silent filtering.** When the Fact-Checker can't confirm a claim, the Writer doesn't drop it — it includes the claim with an explicit confidence caveat. Silently omitting uncertain information erodes trust the first time someone discovers something was missing with no explanation; real intelligence reports communicate certainty levels rather than pretending everything is equally solid.

**A bounded retry loop, not infinite or single-shot.** If the Fact-Checker flags a problem, LangGraph routes the pipeline back to the Researcher automatically — but only up to a configured `max_retries`. Past that, the pipeline proceeds anyway with full transparency about what couldn't be verified, rather than looping forever or failing outright.

**Hand-built MCP server and client.** `crewai-tools`' pre-built search tool had a real, verified import bug in the pinned CrewAI version this project uses. Rather than fight version compatibility, a custom `WebSearchTool` was built directly from CrewAI's `BaseTool`, and later upgraded to a genuine MCP implementation — a standalone server process (`mcp_servers/search_server.py`) speaking the MCP protocol over stdio, and a client tool that launches it as a subprocess. This decouples "how search works" from "how any given agent framework calls it."

**Explicit LLM provider prefixes.** CrewAI routes every model call through LiteLLM, which needs an explicit `provider/model-name` format (e.g. `anthropic/claude-haiku-4-5`) to route correctly — omitting it works by coincidence for OpenAI (LiteLLM's default) but fails outright for other providers. Every agent in this project uses the explicit form for reliability.

---

## Project Structure

```
multi-agent-market-intel/
├── agents/              # The 4 CrewAI agent definitions
│   ├── researcher.py
│   ├── analyst.py
│   ├── fact_checker.py
│   └── writer.py
├── graph/                # LangGraph state machine
│   ├── state.py          # Shared state shape (TypedDict)
│   ├── nodes.py           # Each agent wrapped as a graph node
│   ├── build_graph.py     # Graph assembly, edges, conditional routing
│   └── run_pipeline.py    # Entry point — runs the full pipeline
├── mcp_servers/          # Custom MCP server + client for web search
│   ├── search_server.py
│   └── mcp_search_tool.py
├── k8s/
│   └── cronjob.yaml       # Kubernetes CronJob deployment
├── Dockerfile
├── requirements.txt
└── test_keys.py           # Standalone script to verify all API keys
```

---

## Running It

### Prerequisites
- Python 3.11
- API keys: OpenAI, Anthropic, Google (Gemini), Serper, LangSmith

### Local setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # then fill in your real API keys
python -m graph.run_pipeline
```

### Docker

```bash
docker build -t multi-agent-market-intel .
docker run --env-file .env multi-agent-market-intel
```

### Kubernetes (via Minikube)

```bash
minikube start --driver=docker
minikube image load multi-agent-market-intel
kubectl create secret generic market-intel-secrets --from-env-file=.env
kubectl apply -f k8s/cronjob.yaml

# Trigger a manual run instead of waiting for the schedule:
kubectl create job --from=cronjob/market-intel-report manual-test-run
kubectl logs -f job/manual-test-run
```

The CronJob runs weekly by default (`0 6 * * 1` — every Monday at 6 AM), configurable in `k8s/cronjob.yaml`.

---

## Known Limitations

- **Gemini's free tier** caps at 20 requests/day and 5/minute — sufficient for development and testing, but a production deployment running this on a real schedule would need a paid tier.
- **Claude Haiku 4.5 occasionally deviates from CrewAI's exact expected output format** during long, multi-search fact-checking tasks, triggering automatic reformatting retries within CrewAI itself. This adds latency but doesn't affect correctness.
- **Fact-Checker flags are currently binary** (confirmed/flagged) rather than severity-ranked — a genuine factual error and a minor phrasing nitpick both trigger the same retry behavior. A natural next improvement would be categorizing flag severity.
- **LangSmith tracing via LiteLLM's callback has a benign initialization warning** (`no running event loop`) in synchronous execution contexts — explicitly logged by LiteLLM as non-blocking, and confirmed not to affect trace delivery or pipeline execution.

---

## Author

Vydeesh Mamuduru — [GitHub](https://github.com/vydeesh970)