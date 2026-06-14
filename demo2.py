
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

load_dotenv()

class ChatState(TypedDict):
    user_input: str
    intent: str         # "greeting", "question", or "farewell"
    llm_response: str
    final_output: str

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# ---- NODES ----

def classify_intent_node(state: ChatState) -> dict:
    """Uses the LLM to classify user input into one of three intents."""
    print("[Classifier] Detecting intent...")
    prompt = f"""Classify the following user message into exactly ONE category:
- greeting (hellos, introductions)
- question (they want information)
- farewell (goodbyes, thanks-and-leaving)

Respond with ONLY the category name, nothing else.

User message: "{state['user_input']}"
Category:"""
    
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    print(f"[Classifier] Intent = {intent}")
    return {"intent": intent}

def handle_greeting_node(state: ChatState) -> dict:
    print("[Greeting Handler]")
    return {"llm_response": "Hello! I'm a LangGraph demo bot. Ask me anything."}

def handle_question_node(state: ChatState) -> dict:
    print("[Question Handler] Calling LLM for a real answer...")
    response = llm.invoke(state["user_input"])
    return {"llm_response": response.content}

def handle_farewell_node(state: ChatState) -> dict:
    print("[Farewell Handler]")
    return {"llm_response": "Goodbye! Thanks for testing the graph."}

def format_output_node(state: ChatState) -> dict:
    return {"final_output": f"🤖 {state['llm_response']}"}

# ---- ROUTING FUNCTION ----
# This decides which node to go to next based on state.
# It must return a string that matches a node name (or END).
def route_by_intent(state: ChatState) -> Literal["greeting", "question", "farewell"]:
    intent = state["intent"]
    if "greeting" in intent:
        return "greeting"
    elif "farewell" in intent:
        return "farewell"
    else:
        return "question"  # default fallback

# ---- BUILD GRAPH ----
builder = StateGraph(ChatState)

builder.add_node("classify", classify_intent_node)
builder.add_node("greeting", handle_greeting_node)
builder.add_node("question", handle_question_node)
builder.add_node("farewell", handle_farewell_node)
builder.add_node("format", format_output_node)

# Entry point
builder.add_edge(START, "classify")

# CONDITIONAL EDGE: from "classify", pick a node based on route_by_intent()
builder.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "greeting": "greeting",
        "question": "question",
        "farewell": "farewell",
    }
)

# All three handlers converge on "format"
builder.add_edge("greeting", "format")
builder.add_edge("question", "format")
builder.add_edge("farewell", "format")
builder.add_edge("format", END)

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

# ---- RUN ----
if __name__ == "__main__":
    while True:
        user_input = input("\nYou (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        result = graph.invoke({"user_input": user_input})
        print(result["final_output"])