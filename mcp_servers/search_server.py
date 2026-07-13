"""
The Web Search MCP Server
----------------------------
This is a STANDALONE program - not a class inside your agent code. It runs
as its own separate process and speaks the MCP protocol, which means ANY
MCP-compatible client (your CrewAI agents, Claude Desktop, a completely
different framework someone builds next year) can connect to it and use
its "web_search" capability, without needing custom integration code
written specifically for that client.

This is the architectural upgrade from your hand-rolled WebSearchTool:
that tool was permanently welded to CrewAI's BaseTool interface. This
server doesn't know or care what's connecting to it - it just exposes one
capability over a standard protocol.

HOW THIS RUNS: your agent code will not import this file directly like a
normal Python module. Instead, it launches this file as a SEPARATE
PROCESS (using Python's subprocess mechanism, wrapped by the MCP client
library) and communicates with it over stdin/stdout - this is what
"stdio transport" means. This is genuinely a different execution model
than everything else we've built so far.
"""

import os
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

# FastMCP is the high-level, decorator-based API for building an MCP
# server - similar in spirit to how FastAPI simplifies building a web
# server compared to writing raw HTTP handling code. The string here is
# just this server's name, useful for identifying it in logs.
mcp_server = FastMCP("web-search-server")


@mcp_server.tool()
def web_search(query: str) -> str:
    """
    Searches the web using Google and returns current, real-time results.
    Use this to find up-to-date information such as pricing, recent news,
    or recent feature launches.

    Args:
        query: The search query to look up.

    Returns:
        A formatted string of the top search results, including titles,
        snippets, and source links.
    """
    # This is the EXACT same Serper API logic from your hand-rolled
    # WebSearchTool - the underlying search mechanism hasn't changed at
    # all. What's changed is HOW an agent reaches this code: previously,
    # direct Python function call within the same process. Now, this
    # function is exposed as a formally defined MCP "tool" that gets
    # called over the protocol, by a separate client process.
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload, timeout=15)

    if response.status_code != 200:
        return f"Search failed with status {response.status_code}: {response.text}"

    data = response.json()
    organic_results = data.get("organic", [])[:5]

    if not organic_results:
        return "No search results found for this query."

    formatted = []
    for i, result in enumerate(organic_results, 1):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No description available")
        link = result.get("link", "No link")
        formatted.append(f"{i}. {title}\n   {snippet}\n   Source: {link}")

    return "\n\n".join(formatted)


if __name__ == "__main__":
    # transport="stdio" means this server communicates over standard
    # input/output streams - the same mechanism a command-line program
    # uses to receive input and print output, just repurposed here to
    # carry MCP protocol messages instead of human-readable text. This is
    # the standard, simplest transport for local MCP servers (the
    # alternative, "streamable-http", would make this reachable over a
    # network address instead - useful for remote servers, unnecessary
    # complexity for what we're doing here).
    mcp_server.run(transport="stdio")