***Features***

multiple use cases with the following features:

- LLM agents ( Anthropic & AWS Bedrock )

- RAG hybrid search against a vector DB  with reranking

- RAG non-hybrid search against a vector DB           

- multiple embedding models ( BAAI/HARRIER/QWEN) 

- RAG realized with LangGraph and MCP

- agent orchestration

- HITL

- guardrails

- MLflow   


**Prerequisite**

For direct Anthropic calls the ANTHROPIC_API_KEY env var must be set.

For AWS Bedrock calls you need an AWS acount & AWS_PROFILE must be set.

For HuggingFace model usage ( reranking ) the HF_TOKEN must be set.

( The AWS IAM role behind AWS_PROFILE needs to have a policy assigned to allow for AWS Bedrock functionality to be called. )

In AWS Bedrock you must activate a LLm model. 



**-------------------------------------------------------------------------------------------------------------------**

**rag_hybridsearch_usecase01**

The Qdrant vector database is configured with medicament data ( use  /utils/qdrantDataImporter.py ).

Without using LLM and Langgraph, the database can be directly accessed - 

either via a normal search (  use retrieveNonHybrid ) or via a hybrid search (  use retrieveHybrid ). 

Screenshots of the results under and /data/results/normalSearch_notPrecise.png /data/results/hybridSearch_Precise.png .

The use case allows to test 3 embedding types ( Harrier, Baai and Qwen ). 

Attention database and query must have th same embedding type. 

Use the qdrantDtaImport with the same embedding type as thne one used for searching!

How can it be tested:

*1. Just start the Qdrant vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add MEDICAMENTS documents from data/input/medicamentn to Qdrant  ( embeddings! ):

python  qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 

*2. Execute the query*

python rag_hybridsearch_usecase01.py

**-------------------------------------------------------------------------------------------------------------------**

**rag_mlflow_usecase01**

The Qdrant vector database is configured with medicament data ( use  /utils/qdrantDataImporter.py ).

In this scenario an LLM ( AWS Bedrock ) decides what is to be done to answer the prompty question 

=> therefore a LangGraph graph is created. It contains the famous agnet  loop of reasoning and acting.

Acting means a MCP client : server communication, where the server calls Qdrant.

How can it be tested:

*1. Just start the Qdrant vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add NEWS documents from data/input/news to Qdrant  ( embeddings! ):

python  qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 

*2. Start MLflow server for observability/tracing ( LLM,TOOl,CHAIN,RETRIEVER )*

./mlflow.exe server --port 5000

UI: http://localhost:5000/

*3. Start MCP server*

prerequisite: provide a new port number into server_fastmcp.py @ mcp = FastMCP("demo-server",port=PORTNUMBER)

python fastmcpServer

*4. Execute the query*

python rag_mlflow_usecase01.py

**-------------------------------------------------------------------------------------------------------------------**

**rag_guardrail_usedcase01**

The prompt contains inappropriate language  / an email address.

No LLM is called.

The Langgraph passes control to AWS Guardrail which detects appropriate rule violation.

The rules have been configured in AWS.

The AWS Guardrail answer gets interpreted.

No Agent loop.

**-------------------------------------------------------------------------------------------------------------------**

**rag_guardrail_usedcase02**

The prompt contains inappropriate language  / an email address.

AWS Guardrail is called before the LLM is called.

**-------------------------------------------------------------------------------------------------------------------**

**agentorchestation_usecase01**

Agent hierarchy: a supervisor agents  coordinates 3 low-level agents. 

Orchestration via asyncio features.

No LangGraph.

No  LLM.

**-------------------------------------------------------------------------------------------------------------------**

**hitl_usecase01**

LangGraph-based.

No LLM.

**hitl_usecase02**

LangGraph and LLM.

Tools contain MCP tools and the delete tool.

A user must confirm, that a file is to be deleted.




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