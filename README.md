**Motivation**

This is sandbox project. The goal is to learn essentials of GenAI ( LLM, MCP, RAG, HITL, ... ) 


###
**1st USE CASE 'RAG/MCP/Vector/AWS Bedrock'** 

                        ask agent a question  

                        agent is configured with tools 

                        one tool is a vector search database ( Qdrant )

                        get question answered by using MCP and Qdrant and similiarity search 

                        [ future extensions AWS Guardrail, HITL , ... ]

*0. Prepare environment*

execute the start.sh script ( set ANTHROPIC_API_KEY, AWS_PROFILE )

*1. Start vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add document to Qdrant  ( create Chunks, Embeddings ):

uv run python -m pycharmai.rag.ingestAPI

[ Check content via dashboard: http://localhost:6333/dashboard#/collections ]

*2. Start MCP server*

manually provide a new port number into server_fastmcp.py @ mcp = FastMCP("demo-server",port=PORTNUMBER)

uv run python -m pycharmai.mcp.fastmcpServer

[ check if MCP Server is running: 

Powershell: Test-NetConnection localhost -Port PORTNUMBER

GitBash:    netstat -ano | findstr :PORTNUMBER ]

Potentially use MCP Inspector for  MCP server tests

*3. Execute RAG*

manually set same port number in client.py @  "url": "http://localhost:PORTNUMBER/mcp"

uv run python -m pycharmai.main_rag_mcp_vector_langgraph

Verify, RAG works: 

=> ask "Was ist im Harz passiert", only document news_min.txt includes a reference to the Harz 

=> the answer contains exactly this reference


###
**2nd USE CASE 'Human in the loop (HITL) with Langgraph'** 

uv run python -m main_humanInTheLoop_langgraph


###
**Utilities**

*Document ingestion*

src/pycharmai/rag/ingestAPI.py

*AWS* 

AWS Identität
aws sts get-caller-identity

AWS Profile
$env:AWS_PROFILE="..."

Dependencies
uv add boto3

Bedrock Demo
uv run python src/PyCharmAI/aws/bedrock.py

[Code partially created with help of Gemini & ChatGpt]