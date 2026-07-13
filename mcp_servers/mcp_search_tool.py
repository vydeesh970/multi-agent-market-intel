"""
The MCP-Connected Search Tool
--------------------------------
This replaces the hand-rolled WebSearchTool that lived directly inside
researcher.py and fact_checker.py. It still looks like a normal CrewAI
tool from the agent's perspective (it inherits from BaseTool, same as
before) - but internally, instead of calling the Serper API directly,
it launches your search_server.py as a SEPARATE PROCESS and communicates
with it using the MCP protocol.

WHY THIS INDIRECTION IS WORTH IT: your agent code no longer needs to know
ANYTHING about how search actually works internally (Serper API, request
format, etc.) - it just knows "there's an MCP tool called web_search, and
I can call it." If you swapped search_server.py to use a completely
different search API tomorrow, this file wouldn't need to change at all.
That decoupling is the entire point of MCP.

ASYNC NOTE: the MCP client library is built on Python's asyncio (for
handling the back-and-forth process communication efficiently). CrewAI's
BaseTool._run() method is expected to be a normal, SYNCHRONOUS function
though - so this file includes a small bridge (asyncio.run()) that lets
a synchronous method internally run async code. This is a common, real
pattern you'll see whenever a sync codebase needs to call into an async
library.
"""

import os
import asyncio
from pathlib import Path

from crewai.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# This computes the absolute path to search_server.py, based on THIS
# file's own location, regardless of where the program is run from. This
# matters because we're about to tell Python "launch this file as a
# subprocess" - and subprocess launching needs an exact, reliable path,
# not a guess based on the current working directory.
SEARCH_SERVER_PATH = str(Path(__file__).parent / "search_server.py")


async def _call_mcp_search(query: str) -> str:
    """
    This is the actual async logic that speaks the MCP protocol. It:
      1. Defines HOW to launch the server (as a Python subprocess)
      2. Connects to it over stdio (standard input/output streams)
      3. Opens an MCP "session" - a formal, initialized connection
      4. Calls the 'web_search' tool the server exposes, with our query
      5. Extracts and returns the plain text result
    """
    server_params = StdioServerParameters(
        command="python",  # the executable to run
        args=[SEARCH_SERVER_PATH],  # launches: python search_server.py
        env=os.environ.copy(),  # pass through our .env variables (like
                                  # SERPER_API_KEY) to the subprocess -
                                  # without this, the server wouldn't see
                                  # our API keys, since it's a separate
                                  # process with its own environment.
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # initialize() performs the MCP protocol's opening handshake
            # - the client and server confirm they understand each other
            # before any real work happens, similar in spirit to how a
            # web browser and server negotiate a connection before
            # exchanging actual page content.
            await session.initialize()

            result = await session.call_tool("web_search", {"query": query})

            # The result contains a list of "content" blocks (MCP
            # supports multiple content types - text, images, etc.). Our
            # server only ever returns plain text, so we extract that.
            text_parts = [
                block.text for block in result.content if hasattr(block, "text")
            ]
            return "\n".join(text_parts) if text_parts else "No results returned."


class MCPWebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Searches the web using Google and returns current, real-time "
        "results. Use this whenever you need up-to-date information such "
        "as current pricing, recent news, or recent feature launches - "
        "do not rely on your training data for anything time-sensitive."
    )

    def _run(self, query: str) -> str:
        # asyncio.run() is the bridge mentioned above: it takes our async
        # _call_mcp_search function and runs it to completion inside this
        # normal, synchronous method, blocking until the result is ready
        # - exactly what CrewAI expects a tool's _run() to do.
        return asyncio.run(_call_mcp_search(query))