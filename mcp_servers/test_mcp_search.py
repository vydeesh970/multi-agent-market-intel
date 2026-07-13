"""
A quick standalone test of the MCP search tool - confirms the whole
client-server round trip works BEFORE we wire it into a real agent. This
isolates one new moving part (MCP) at a time, instead of debugging MCP
and agent behavior simultaneously.
"""

from dotenv import load_dotenv
load_dotenv()

from mcp_search_tool import MCPWebSearchTool

if __name__ == "__main__":
    tool = MCPWebSearchTool()

    print("Launching MCP server and sending a test search query...")
    print("(This spins up search_server.py as a subprocess behind the scenes)\n")

    result = tool._run(query="GitHub Copilot pricing")

    print("=" * 70)
    print("RESULT FROM MCP SERVER:")
    print("=" * 70)
    print(result)