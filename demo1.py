from typing import TypedDict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

load_dotenv()  # Loads OPENAI_API_KEY from .env

# 1. DEFINE THE STATE
# This is the "whiteboard" passed between nodes.
# Every node reads from it and returns updates to it.
class ChatState(TypedDict):
    user_input: str
    llm_response: str
    final_output: str

# 2. INITIALIZE THE LLM (shared across nodes)
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# 3. DEFINE NODES
# A node is just a function: (state) -> dict of updates
def call_llm_node(state: ChatState) -> dict:
    """Takes user_input from state, asks the LLM, returns the response."""
    print("[Node A] Calling LLM...")
    response = llm.invoke(state["user_input"])
    # Return ONLY the keys you want to update in state
    return {"llm_response": response.content}

def format_output_node(state: ChatState) -> dict:
    """Formats the LLM response for display."""
    print("[Node B] Formatting output...")
    formatted = f"🤖 Assistant: {state['llm_response']}"
    return {"final_output": formatted}

# 4. BUILD THE GRAPH
builder = StateGraph(ChatState)

# Add nodes (give each a name + the function)
builder.add_node("call_llm", call_llm_node)
builder.add_node("format_output", format_output_node)

# Add edges (define the flow)
builder.add_edge(START, "call_llm")        # Start → call_llm
builder.add_edge("call_llm", "format_output")  # call_llm → format_output
builder.add_edge("format_output", END)     # format_output → End

# 5. COMPILE
graph = builder.compile()
# Save a PNG of the graph structure
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph diagram saved to graph.png")
except Exception as e:
    # Falls back to text representation if rendering fails
    print(graph.get_graph().draw_mermaid())

# 6. RUN IT
if __name__ == "__main__":
    user_question = input("You: ")
    
    # Invoke with initial state (only fields you want to set at the start)
    result = graph.invoke({"user_input": user_question})
    
    print("\n--- Final State ---")
    print(result["final_output"])