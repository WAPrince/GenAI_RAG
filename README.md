***Motivation***

Features

-LLM agents ( Anthropic & AWS Bedrock )

-agent orchestration, 

-LangGraph,

-RAG, MCP, VectorDB,

-multiple embedding models ( BAAI/bge-small-en-v1.5, microsoft/harrier-oss-v1-0.6b, Qwen/Qwen3-Embedding-8B)

-HITL, 

-guardrails,

-MLflow   



**Prerequisite**

For direct Anthropic calls the ANTHROPIC_API_KEY env var must be set.

For AWS Bedrock calls you need an AWS acount & AWS_PROFILE must be set.

( The AWS IAM role behind AWS_PROFILE needs to have a policy assigned to allow for AWS Bedrock functionality to be called. )

In Bedrock you must activate a LLm model. 

execute the start.sh script ( **`set ANTHROPIC_API_KEY, AWS_PROFILE env vars`** )



**--------------------------------------------------------------------------------------------------------------------**

**1st USE CASE 'Simple Anthropic chat'**
                            
uv run python -m main01_llmAnthropic



**--------------------------------------------------------------------------------------------------------------------**

**2nd USE CASE 'Human in the loop (HITL) with Langgraph only'** 

                        agent asks user two times for approval

                        realise a generic interrupt handling ( reusable )

uv run python -m main02_langgraph_hitl



**--------------------------------------------------------------------------------------------------------------------**

**3rd USE CASE 'Human in the loop (HITL) with llm ( Anthropic via AWS Bedrock ), tool and Langgraph'** 
                        
                        ask agent to delete a file using a tool

                        agent requires human approval 

                        simple interrupt handling

uv run python -m main03_llmBedrock_langgraph_tool_htil



**--------------------------------------------------------------------------------------------------------------------**

**4 USE CASES **
'Retrieval augmented generation' with llm ( Anthropic via AWS Bedrock ), rag ( mcp, qdrant vector ) and Langgraph'** 

                        before: a document has been injected into the vector search database
                                the agent is configured with one tool that equals a MCP client 
                        
                        then: ask the agent a question 

                        the agent needs its tool

                        the tool calls the MCP client => MCP server

                        the MCP server delegates the query to a vector search database ( Qdrant Docker container )

                        execution of a similiarity search using the query embedding and the chunk embeddings

                        [ future extensions AWS Guardrail, HITL , ... ]


**----------------------**

*1. Start MLflow server for observability/tracing ( LLM,TOOl,CHAIN,RETRIEVER )*

./mlflow.exe server --port 5000

UI: http://localhost:5000/



**----------------------**

*2. Start vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add document to Qdrant  ( create Chunks, Embeddings ):

python  qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 


**----------------------**

*3. Start MCP server*

prerequisite: provide a new port number into server_fastmcp.py @ mcp = FastMCP("demo-server",port=PORTNUMBER)

python fastmcpServer


**----------------------**

*4.1 Execute RAG with MLflow*

python main04.1_llmBedrock_ragMcpVector_hitl_langgraph.py

User Question:  "Was ist in Hannover passiert? Rufe die Vektordatenbank auf!"

Answer: 



*4.2 Attempt to delete a file*

python main04.2_llmBedrock_ragMcpVector_langgraph_MLflow

Verify, human in the loop ( look at shell! ) is executed.



**--------------------------------------------------------------------------------------------------------------------**


**5th USE CASE 'A super agent orchestrates three worker agents - asynchronous handling'**

                            a super agent ask three worker agents and asynchronously incorporates their answer

uv run python -m main05_agent_orchestrator


**--------------------------------------------------------------------------------------------------------------------**

**6th USE CASE 'AWS Guardrails integration'**
        
                            user request containing inappropriate language or PII will be incepted by AWS guardrail and
                            will not be forwarded to the LLM

uv run python -m main06_llmBedrock_ragMacVector_guardrail_langgraph


**--------------------------------------------------------------------------------------------------------------------**

**7th USE CASE 'FastAPI Streaming'**
=> see folder src/pycharmai/demo/streaming and data/FastAPI_Streaming.png


**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**

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