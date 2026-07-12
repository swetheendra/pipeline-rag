from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_tavily import TavilySearch
from pinecone import ServerlessSpec, Pinecone
from langchain_core.stores import InMemoryStore
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import MistralAIEmbeddings
from typing import TypedDict, Annotated, Literal, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import json
import asyncio
import streamlit as st

# Bridge Streamlit secrets into environment variables so that both this code
# (os.environ.get) and LangChain tools (which read keys like TAVILY_API_KEY
# directly from the environment) can find them. Works on Streamlit Cloud and
# locally via .streamlit/secrets.toml.
for _key in ("MISTRAL_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        # No secrets file present (e.g. pure env-var deployment); fall back to
        # whatever is already in os.environ.
        break

# Fail loudly but gracefully (in-app) instead of crashing the process at import
# time. A missing key would otherwise make Pinecone()/ChatMistralAI() raise
# during module load, which Streamlit Cloud reports as an opaque "spawn error".
_required = ("MISTRAL_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY")
_missing = [k for k in _required if not os.environ.get(k)]
if _missing:
    st.error(
        "Missing required secrets: "
        + ", ".join(_missing)
        + ". Add them under App → Settings → Secrets on Streamlit Cloud, "
        "or in a local .streamlit/secrets.toml file, then rerun."
    )
    st.stop()


def reduce_context(left: list[str], right: list[str]) -> list[str]:
    return right


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: Annotated[list[str], reduce_context]
    is_faithful: Literal["yes", "no", "not_evaluated"]
    loop_count: int


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question, score 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in / supported by facts in context, score 'yes' or 'no'")


def create_pinecone_index():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "rag-store-mistral"
    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)


@st.cache_resource(show_spinner="Loading embedding model and building the RAG pipeline (first run only)...")
def build_graph():
    """Build heavy resources and the compiled LangGraph exactly once.

    Everything that downloads a model, hits Pinecone, or re-ingests the PDF
    lives here so it runs a single time per container instead of on every
    Streamlit rerun. Node functions are nested so they close over these
    resources instead of relying on module globals.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the bulletproof absolute path to the PDF
    pdf_path = os.path.join(current_dir, "2025_AnnualReport.pdf")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # 1. Define the Wrapper Class
    class SanitizedMistralEmbeddings(Embeddings):
        # mistral-embed rejects large requests with a 400; keep each request small.
        BATCH_SIZE = 32

        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # Clean each text but PRESERVE list length (1:1 with input) so the
            # returned embeddings stay aligned with Pinecone ids/metadatas.
            cleaned_texts = []
            for t in texts:
                cleaned = str(t).replace("\x00", "").strip() if t is not None else ""
                # Replace empties with a fallback token instead of dropping them,
                # otherwise the embedding count won't match the document count.
                cleaned_texts.append(cleaned if cleaned else "fallback_text")

            # Batch requests so we never exceed mistral-embed's per-request limit.
            embeddings: List[List[float]] = []
            for i in range(0, len(cleaned_texts), self.BATCH_SIZE):
                batch = cleaned_texts[i:i + self.BATCH_SIZE]
                embeddings.extend(self.model.embed_documents(batch))
            return embeddings

        def embed_query(self, text: str) -> List[float]:
            cleaned = str(text).replace("\x00", "").strip() or "fallback_text"
            return self.model.embed_query(cleaned)

    # 2. Instantiate the raw model with your specific environment key
    raw_mistral = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ.get("MISTRAL_KEY"),
    )

    # 3. Wrap it! This 'embeddings' variable goes right into your Pinecone/Retriever setup
    embeddings = SanitizedMistralEmbeddings(model=raw_mistral)

    pinecone_index = create_pinecone_index()
    docstore = InMemoryStore()
    store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key="text")

    retriever = ParentDocumentRetriever(
        vectorstore=store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 1. Clean and validate documents before passing to ParentDocumentRetriever
    sanitized_documents = []

    for doc in documents:
        # Remove null bytes and standardize whitespaces
        text = doc.page_content.replace("\x00", "").strip()
        
        # Filter out empty or pure whitespace chunks which trigger Mistral 400 errors
        if text:
            doc.page_content = text
            sanitized_documents.append(doc)

    # Create a safe function wrapper to intercept and sanitize chunks
    def safe_add_documents(documents, **kwargs):
        sanitized = []
        for doc in documents:
            cleaned_text = doc.page_content.replace("\x00", "").strip()
            # Only allow chunks that actually contain readable characters
            if cleaned_text:
                doc.page_content = cleaned_text
                sanitized.append(doc)
        # Forward the clean chunks to the real Pinecone backend
        return original_add_documents(sanitized, **kwargs)

    # Intercept the vectorstore method
    original_add_documents = retriever.vectorstore.add_documents
    retriever.vectorstore.add_documents = safe_add_documents

    # Now run your addition safely
    retriever.add_documents(sanitized_documents, ids=None)

    # Mistral's free tier allows ~1 request/second. Throttle proactively so the
    # call-heavy graph (per-doc grading + synthesis + rewrite loops) does not
    # burst past the limit and get a 429. This blocks locally instead of failing.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.environ.get("MISTRAL_KEY"),
        temperature=0,
        max_retries=3,
        rate_limiter=rate_limiter,
    )
    llm_with_grading = llm.with_structured_output(GradeDocuments)
    llm_with_hallucination = llm.with_structured_output(GradeHallucinations)

    def retrieve_from_pinecone(state: AgentState):
        user_query = state["messages"][-1].content
        retrieved_docs = retriever.invoke(user_query)
        contents = [doc.page_content for doc in retrieved_docs]
        return {"context": contents}

    def grade_documents(state: AgentState):
        user_query = state["messages"][-1].content
        context = state["context"]

        if not context:
            return {"context": []}

        filtered_context = []
        for doc in context:
            grader_prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
            \nDocument: {doc} \nUser Question: {user_query}
            \nDetermine if the document contains semantic keywords or answers to the question."""

            res = llm_with_grading.invoke(grader_prompt)
            if res.binary_score == "yes":
                filtered_context.append(doc)

        return {"context": filtered_context}

    def synthesis_node(state: AgentState):
        user_query = state["messages"][-1].content
        context = state["context"]

        if not context or len(context) == 0:
            return {
                "messages": [AIMessage(content="I'm sorry, but the retrieved documents do not contain information relevant to your request.")]
            }

        combined_context = "\n\n".join(context)

        system_prompt = f"""You are a precise, literal data extraction engine. Your task is to answer the user's query using ONLY the explicitly stated facts in the provided Context. 
            
            CRITICAL SAFETY RULES:
            1. Grounding: Do not add background context, historical timelines, tech specs, or statistics unless they are written verbatim in the context block below. 
            2. No Extrapolation: If the context says a protocol "helps limit metadata," do not explain *how* it limits metadata using your own knowledge of network packets.
            3. Missing Information: If the context is insufficient to fully answer the comparison, state clearly what the context *does* provide, and note that the remaining details are missing from the source material.
            
            Context:
            {combined_context}
            
            User Query: {user_query}
            Answer:"""

        response = llm.invoke(system_prompt)
        return {"messages": response}

    def query_rewriter_node(state: AgentState):
        # CRITICAL: Always pull the first message (the user's actual question),
        # not the last message which might be a failed synthesis or previous rewrite!
        original_user_query = state["messages"][0].content

        prompt = f"Optimize this user query for a Google search. Return ONLY the search terms. Do not include introductory text, explanations, or multiple choices:\n\n{original_user_query}"

        response = llm.invoke(prompt)

        # Strip any markdown backticks or text wraps the LLM might have added
        clean_query = response.content.replace("`", "").replace("*", "").strip()
        new_count = state.get("loop_count", 0) + 1
        return {"messages": [HumanMessage(content=clean_query)], "loop_count": new_count}

    def web_search_node(state: AgentState):
        raw_query = state["messages"][-1].content
        clean_query = raw_query.replace("**", "").replace('"', '').strip()

        web_search_tool = TavilySearchResults(max_results=3)
        results = web_search_tool.invoke({"query": clean_query[:380]})

        # --- DEFENSIVE PARSING START ---
        # If the tool returned a stringified JSON array, parse it back to a list
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                # Fallback if it's a plain unstructured text string instead of JSON
                return {"context": [results]}
        # --- DEFENSIVE PARSING END ---

        # Securely extract content now that results is guaranteed to be a list of dicts
        extracted_contents = [item["content"] for item in results if isinstance(item, dict) and "content" in item]

        return {"context": extracted_contents}

    def grade_generation_node(state: AgentState):
        context = " ".join(state["context"])
        answer = state["messages"][-1].content
        prompt = f"""You are an auditor verifying claims. 
            Source Context: {context}
            Generated Answer: {answer}
            
            Is every single claim in the Generated Answer explicitly supported by the Source Context? 
            Respond with exactly 'yes' or 'no'."""

        res = llm_with_hallucination.invoke(prompt)
        return {"is_faithful": res.binary_score}

    def grade_doc_router(state: AgentState):
        if not state["context"]:
            return "rewrite"
        else:
            return "synthesis"

    def hallucinating_router(state: AgentState):
        loops = state.get("loop_count", 0)
        if state.get("is_faithful") == "yes" or loops >= 5:
            return "finish"
        return "rewrite"

    grader = StateGraph(AgentState)
    grader.add_node("retrieve_node", retrieve_from_pinecone)
    grader.add_node("grade_documents_node", grade_documents)
    grader.add_node("synthesis", synthesis_node)
    grader.add_node("query_rewriter_node", query_rewriter_node)
    grader.add_node("web_search_node", web_search_node)
    grader.add_node("grade_generation", grade_generation_node)

    grader.add_edge(START, "retrieve_node")
    grader.add_edge("retrieve_node", "grade_documents_node")

    grader.add_conditional_edges("grade_documents_node", grade_doc_router, {
        "rewrite": "query_rewriter_node",
        "synthesis": "synthesis"
    })

    grader.add_edge("query_rewriter_node", "web_search_node")
    grader.add_edge("web_search_node", "grade_documents_node")

    grader.add_edge("synthesis", "grade_generation")

    grader.add_conditional_edges("grade_generation", hallucinating_router, {
        "finish": END,
        "rewrite": "query_rewriter_node"
    })

    return grader.compile()


st.title("📉 RAG Pipeline")
user_query = st.text_input("Enter your question:", value="")

if st.button("Run Pipeline") and user_query:

    graph = build_graph()

    status_placeholder = st.empty()
    output_placeholder = st.empty()

    async def run_pipeline_stream():
        inputs = {"messages": [HumanMessage(content=user_query)]}

        with status_placeholder.status("🤖 Initializing Graph Pipeline...", expanded=True) as status:

            async for chunk in graph.astream(inputs, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    status.update(label=f"📍 Currently Executing: **{node_name}**")

                    st.write(f"✓ **{node_name}** complete.")
                    with st.expander(f"View State Delta for {node_name}"):
                        st.json(state_update)

                    if node_name == "synthesis":
                        messages_data = state_update["messages"]
                        if isinstance(messages_data, list):
                            st.session_state["final_answer"] = messages_data[-1].content
                        else:
                            st.session_state["final_answer"] = messages_data.content

            status.update(label="✅ Pipeline Execution Complete!", state="complete", expanded=False)

    if "final_answer" in st.session_state:
        del st.session_state["final_answer"]

    asyncio.run(run_pipeline_stream())

    if "final_answer" in st.session_state:
        output_placeholder.subheader("📝 Final Answer")
        output_placeholder.write(st.session_state["final_answer"])
