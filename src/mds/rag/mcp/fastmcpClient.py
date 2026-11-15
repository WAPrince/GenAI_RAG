from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_SERVERS = {
    "medicaments": {
        "transport": "streamable-http",
        "url": "http://localhost:3006/mcp",
    },
    "medicaments__dense_sparse": {
        "transport": "streamable-http",
        "url": "http://localhost:3006/mcp",
    }#,
    #"news": {
    #    "transport": "streamable-http",
    #    "url": "http://localhost:3006/mcp",
    #},
    #"news__dense_sparse": {
    #    "transport": "streamable-http",
    #    "url": "http://localhost:3006/mcp",
    #}

}

def create_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(MCP_SERVERS)