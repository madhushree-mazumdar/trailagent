"""
langgraph_app.py
LangGraph application setup for TrailAgent
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from typing import TypedDict, List, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import os

# Set up NVIDIA API key (ensure this is set securely in production)
if not os.environ.get("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = "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"

# Chat model
chat_model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Llama Guard model for safety checks
llama_guard_model = ChatNVIDIA(model="meta/llama-guard-4-12b", temperature=0.0)

# Embeddings model
embeddings_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# Vector store retriever (Qdrant)
def get_vectorstore_retriever():
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="nps",
        embedding=embeddings_model,
        url="https://e08da72a-93d5-4d9b-9ba9-b49759659b78.us-west-2-0.aws.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.YTnvpCTAvZrM7lW-dOvvFq54IWyFW2bpi-RVmKP2Nak",
        vector_name="nps-dense"
    )
    return vector_store.as_retriever()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: Tracks message history.
        documents: List of retrieved documents.
        safety_flag: Will be 'SAFE', 'UNSAFE_INPUT', or 'UNSAFE_OUTPUT'
    """
    messages: Annotated[list[BaseMessage], add_messages]
    documents: List[Document]
    safety_flag: str
    unsafe_category: str

class TrailAgentLangGraph:
    def __init__(self):
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model
        self.retriever = get_vectorstore_retriever()
        self.categories = {
            "s1": "Violent Crimes",
            "s2": "Non-Violent Crimes",
            "s3": "Sex Crimes",
            "s4": "Child Exploitation",
            "s5": "Defamation",
            "s6": "Specialized Advice",
            "s7": "Privacy",
            "s10": "Hate Speech",
            "s11": "Suicide/Self-Harm",
            "s13": "Elections"
        }
    
    def check_safety(self, llm_client, conversation_messages):
        # Llama Guard usually works by assessing the turn-by-turn conversation history.
        # It returns a response like "safe" or "unsafe" with a violation category.
        message = [HumanMessage(content= conversation_messages[-1].content)]
        response = llm_client.invoke(message)
        lines = response.content.strip().split("\n")
        status = lines[0].lower() # 'safe' or 'unsafe'
        print("Guardrail response: ", status)
        return {"status": status, "category": lines[1].lower()}

    # 1. Input Gate Node
    def check_input_safety(self, state: GraphState) -> dict:
        # Check the latest message (which is the user's input at the start)
        safety_result = self.check_safety(llama_guard_model, state["messages"])
        if safety_result["status"] == "unsafe":
            return {"safety_flag": "UNSAFE_INPUT", "unsafe_category": safety_result["category"]}
        return {"safety_flag": "SAFE"}


    # 2. Output Gate Node
    def check_output_safety(self, state: GraphState) -> dict:
        # Check the latest message (which is the AI's response)
        safety_result = self.check_safety(llama_guard_model, state["messages"])
        if safety_result["status"] == "unsafe":
            return {"safety_flag": "UNSAFE_OUTPUT", "unsafe_category": safety_result["category"]}
        # If the output is safe, the process is complete
        return {"safety_flag": "SAFE"}

    def route_to_next_step(self, state: GraphState, current_node: str = None) -> str:
        """Routes execution based on the safety flag and current node."""
        # For input_guard, SAFE should go to 'retrieve'.
        # For output_guard, SAFE should go to END.
        if state["safety_flag"] == "UNSAFE_INPUT":
            return "halt_process"
        elif state["safety_flag"] == "UNSAFE_OUTPUT":
            return "halt_process"
        elif state["safety_flag"] == "SAFE":
            if current_node == "input_guard":
                return "retrieve"
            elif current_node == "output_guard":
                return "SAFE"
        # Fallback for unexpected cases
        raise ValueError(f"Unknown routing for node {current_node} with flag {state['safety_flag']}")

    def halt_process(self, state: GraphState) -> dict:
        if state["safety_flag"] == "UNSAFE_INPUT":
            msg = "I cannot process that request due to safety policy violations: " + self.categories[state["unsafe_category"]]
        else: # UNSAFE_OUTPUT
            msg = "The generated response was flagged as potentially unsafe and has been blocked: " + self.categories[state["unsafe_category"]]
        return {"messages": [AIMessage(content=msg)]}

    def retrieve(self, state):
        """Retrieves documents from the vector store."""
        print("---RETRIEVING DOCUMENTS---")
        question = state["messages"][-1].content
        documents = self.retriever.invoke(question)
        #print(documents)
        return {"documents": documents}

    def generate(self, state):
        """Generates an answer using the LLM and retrieved documents."""
        print("---GENERATING ANSWER---")
        question = state["messages"][-1].content
        documents = state["documents"]
        #print(documents)

        # Prompt for RAG
        prompt = ChatPromptTemplate.from_template(
            """You are the National Park AI Assistant, specialized in answering questions about National Parks in California.
            You operate using retrieval-augmented generation (RAG) and rely exclusively on the information retrieved from the vector database provided to you.
            Your goal is to provide accurate, concise, and contextually relevant answers based solely on the retrieved documents.

            ────────────────────────────
            CORE INSTRUCTIONS (RAG-BOUND)
            ────────────────────────────
            Answer only the specific question asked.
            Do not add additional facts, tips, or suggestions unless the user explicitly asks for them.
            Do not generate or infer facts that are not grounded in the retrieved documents.
            No assumptions.
            No external knowledge.
            No filling in the gaps.
            If the user asks about topics outside California National Parks, respond with:
            “I can only answer questions about National Parks in California.”
            Keep answers concise, focused, and factual.
            If the user asks for opinions, clarify that you do not have personal opinions but can provide factual information if available in the context.

            ────────────────────────────
            HALLUCINATION PREVENTION
            ────────────────────────────
            If the context does not mention a trail, wildlife species, date, rule, or any fact → do NOT invent it.
            Do not reference real-time conditions such as weather, closures, or crowd levels.

            Context: {context}
            Question: {question}
            """
        )

        # RAG chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {"context": lambda x: "\n".join(doc.page_content for doc in x['documents']),
            "question": RunnablePassthrough()}
            | prompt
            | chat_model
        )

        # Run the generation
        response = rag_chain.invoke({"documents": documents, "question": question})
        print(response.content)

        return {"messages": [AIMessage(content=response.content)], "documents": documents}
    
    def generate_response(self, question: str) -> str:
        workflow = StateGraph(GraphState)

        workflow.add_node("halt_process", self.halt_process)
        workflow.add_node("retrieve", self.retrieve)  # Retrieves documents
        workflow.add_node("generate", self.generate)  # Generates answer
        workflow.add_node("input_guard", self.check_input_safety)
        workflow.add_node("output_guard", self.check_output_safety)
        

        workflow.add_conditional_edges(
            "input_guard",
            lambda state: self.route_to_next_step(state, current_node="input_guard"),
            {
                "retrieve": "retrieve",
                "halt_process": "halt_process"
            }
        )

        # --- RAG and Output Gate Logic ---
        # RAG always proceeds to the output check
        workflow.add_edge("generate", "output_guard")

        # After output_guard, check the flag and route
        workflow.add_conditional_edges(
            "output_guard",
            lambda state: self.route_to_next_step(state, current_node="output_guard"),
            {
                "halt_process": "halt_process",
                "SAFE": END # If safe, the agent is done
            }
        )

        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "output_guard")
        workflow.add_edge("halt_process", END)
        workflow.set_entry_point("input_guard")

        # Compile the graph
        app = workflow.compile()
        message = [HumanMessage(content=question)]
        final_state = app.invoke({"messages": message})

        print("\n\n--- FINAL ANSWER ---")
        print(final_state["messages"][-1].content)
        return final_state["messages"][-1].content