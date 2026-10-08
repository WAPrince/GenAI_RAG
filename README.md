**Motivation**

This is a sandbox project. The goal is to learn essentials of GenAI ( LLM, MCP, RAG, HITL, ... ) 


**Prerequisite**

execute the start.sh script ( set ANTHROPIC_API_KEY, AWS_PROFILE env vars )


###
**1st USE CASE 'Mock an agent tool answer'**
    
                        agent has a mocked tool answer

uv run python -m main01_mocked_tool


###
**2nd USE CASE 'Human in the loop (HITL) with Langgraph only'** 

                        agent asks user two times for approval

                        realise a generic interrupt handling ( reusable )

uv run python -m main02_hitl_langgraph


###
**3rd USE CASE 'Human in the loop (HITL) with llm ( Anthropic via AWS Bedrock ), tool and Langgraph'** 
                        
                        ask agent to delete a file using a tool

                        agent requires human approval 

                        simple interrupt handling

uv run python -m main03_hitl_llm_tool_langgraph


###
**4th USE CASE 'Retrieval augmented generation' with llm ( Anthropic via AWS Bedrock ), rag ( mcp, vector ) and Langgraph'** 

                        before: a document has been injected into the vector search database
                                the agent is configured with one tool that equals a MCP client 
                        
                        then: ask the agent a question 

                        the agent needs its tool

                        the tool calls the MCP client => MCP server

                        the MCP server delegates the query to a vector search database ( Qdrant Docker container )

                        execution of a similiarity search using the query embedding and the chunk embeddings

                        [ future extensions AWS Guardrail, HITL , ... ]

*1. Start vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add document to Qdrant  ( create Chunks, Embeddings ):

uv run python -m pycharmai.rag.ingestAPI

[ Check content via dashboard: http://localhost:6333/dashboard#/collections ]

*2. Start MCP server*

prerequisite: provide a new port number into server_fastmcp.py @ mcp = FastMCP("demo-server",port=PORTNUMBER)

uv run python -m pycharmai.mcp.fastmcpServer

[ check if MCP Server is running: 

Powershell: Test-NetConnection localhost -Port PORTNUMBER

GitBash:    netstat -ano | findstr :PORTNUMBER ]

Potentially use MCP Inspector for  MCP server tests

*3. Execute RAG*

prerequisite: manually set same port number in client.py @  "url": "http://localhost:PORTNUMBER/mcp"

uv run python -m pycharmai.main04_llm_rag_mcp_vector_langgraph

Verify, RAG works: 

=> ask "Was ist im Harz passiert", only document news_min.txt includes a reference to the Harz 

=> the answer contains exactly this reference


###
**5th USE CASE 'A super agent orchestrates three worker agents - asynchronous handling'**

                            a super agent ask three worker agents and asynchronously incorporates their answer

uv run python -m main05_agent_orchestrator


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