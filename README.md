***FEATURES***

- LLM agents ( Anthropic & AWS Bedrock )

- RAG hybrid search against a vector DB  with reranking

- RAG non-hybrid search against a vector DB           

- multiple embedding models ( BAAI/HARRIER/QWEN ) 

- RAG realized with LangGraph and MCP

- agent orchestration

- HITL

- guardrails

- MLflow   

**RECOMMENDATION**

test rag_hybridsearch_withAnthropic_hitlQuestionLoop_usecase03.2 !

**PREREQUISITES**

For direct Anthropic calls the ANTHROPIC_API_KEY env var must be set.

For AWS Bedrock calls you need an AWS acount & AWS_PROFILE must be set. The AWS IAM role behind AWS_PROFILE needs to have a policy assigned to allow for AWS Bedrock functionality to be called. In AWS Bedrock you must activate an LLm model. 

For HuggingFace model usage ( reranking ) the HF_TOKEN must be set.


**USE CASES**

**-------------------------------------------------------------------------------------------------------------------**

**rag_hybridsearch_withoutAgent_usecase01 ( without LLM and without MCP )**

The Qdrant vector database is configured with MEDICAMENTS data ( use  /utils/qdrantDataImporter.py ).

Without using LLM, MCP and Langgraph, the database can be directly accessed - 

either via a normal search or via a hybrid search ( with embedding types Harrier, Baai  ). 

Hybrid search: 1st step: a sparse and a dense search are being executed. In the 2nd step both results are being reranked using a Hugging Face model.

Screenshots of the results under and /data/results/normalSearch_notPrecise.png /data/results/hybridSearch_Precise.png .

Attention: database and query must have the same embedding type. Use the qdrantDataImport with the same embedding type as the one used for searching!


How can it be tested:


*1. Just start the Qdrant vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add MEDICAMENTS documents from data/input/medicament to Qdrant  ( embeddings! ):

python  qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 

Collections 'medicaments' and 'medicaments__dense_sparse' will be created!


*2. Execute the query*

python rag_hybridsearch_usecase01.py

**-------------------------------------------------------------------------------------------------------------------**

**rag_hybridsearch_withAnthropic_hitlQuestionLoop_usecase03.2 ( with LLM and with MCP - with HITL for question loop )**

( Log output of this usecase: => data/results/logs/baai_hyb_llm_loop.txt )

In this scenario an LLM decides what is to be done to answer the prompt question 

=> therefore a LangGraph graph is created. It contains the famous agent loop of reasoning and acting.

Acting means an MCP client : server communication, where the server calls Qdrant.

The Qdrant vector database is configured with MEDICAMENTS data ( use  /utils/qdrantDataImporter.py ).

How can it be tested:

*1. Just start the Qdrant vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add the MEDICAMENTS document from data/input/medicemtns to Qdrant  ( embeddings! ):

cd src
src> ..\.venv\Scripts\python.exe -m utils.qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 

Collections 'medicaments' and 'medicaments__dense_sparse' will be created!


*2. Start LangFuse*

(   install:  git clone https://github.com/langfuse/langfuse.git &   dock-compose up -d )

UI: http://localhost:3000/


*3.[Optional] Start MLflow server for observability/tracing ( LLM,TOOl,CHAIN,RETRIEVER )*

./mlflow.exe server --port 5000

UI: http://localhost:5000/


*4. Start MCP server*

cd src
src> ..\.venv\Scripts\python.exe -m mcp_server


*5. Execute the query and interact with the agent*

cd src

Attention: activate the prompts you want to test!

src> ..\.venv\Scripts\python.exe -m rag_hybridsearch_withAnthropic_hitlQuestionLoop_usecase03.2.py

Type-in questions like: 

Welche Nebenwirkung hat Doxagamma? Nutze Datenbank medicaments__dense_sparse! Nenne deine Quellen!

Welche Nebenwirkung hat Amlodipin? Nutze Datenbank medicaments__dense_sparse! Nenne deine Quellen!
                        
quit

**-------------------------------------------------------------------------------------------------------------------**

**rag_hybridsearch_withAnthropic_usecase03.1 rag_hybridsearch_withBedrock_usecase02 ( with LLM and with MCP - without HITL )**

In this scenario an LLM decides what is to be done to answer the prompt question 

=> therefore a LangGraph graph is created. It contains the famous agent loop of reasoning and acting.

Acting means an MCP client : server communication, where the server calls Qdrant.

The Qdrant vector database is configured with MEDICAMENTS data ( use  /utils/qdrantDataImporter.py ).


How can it be tested:

*1. Just start the Qdrant vector database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add the MEDICAMENTS document from data/input/medicemtns to Qdrant  ( embeddings! ):

cd src
src> ..\.venv\Scripts\python.exe -m utils.qdrantDataImporter

UI: http://localhost:6333/dashboard#/collections 

Collections 'medicaments' and 'medicaments__dense_sparse' will be created!


*2. Start MLflow server for observability/tracing ( LLM,TOOl,CHAIN,RETRIEVER )*

./mlflow.exe server --port 5000

UI: http://localhost:5000/


*3. Start MCP server*

cd src
src> ..\.venv\Scripts\python.exe -m mcp_server



*4. Execute the query*
cd src
Attention: activate the prompts you want to test!
src> ..\.venv\Scripts\python.exe -m rag_hybridsearch_withAnthropic_usecase03.1.py

src> ..\.venv\Scripts\python.exe -m rag_hybridsearch_withBedrock_usecase02.py

Logs of successful runs are under ./data/results/logs.

**rag_guardrail_usedcase01**

The prompt contains inappropriate language  / an email address.

No LLM is called.

The LangGraph passes control to AWS Guardrail which detects appropriate rule violation.

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