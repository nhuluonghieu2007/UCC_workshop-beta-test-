from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ─────────────────────────────────────────
# 1. DEFINE YOUR TOOLS
# ─────────────────────────────────────────
# The @tool decorator does three things:
#   - makes the function callable by the LLM
#   - uses the function name as the tool name
#   - uses the docstring as the tool description (the LLM reads this!)

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together. Use this when the user asks for addition."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers. Use this when the user asks for multiplication."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when the user asks about weather."""
    # Fake data — in a real app you'd call a weather API here
    fake_weather = {
        "hanoi":   "32°C, humid, partly cloudy",
        "london":  "14°C, overcast, light rain",
        "tokyo":   "22°C, clear skies",
    }
    return fake_weather.get(city.lower(), f"No weather data found for '{city}'.")

# Collect tools into a list — this list is used in two places below
tools = [add_numbers, multiply_numbers, get_weather]


# ─────────────────────────────────────────
# 2. SET UP THE LLM WITH TOOLS BOUND
# ─────────────────────────────────────────
# bind_tools() tells the LLM: "you are allowed to call these tools"
# Without this, the LLM will just answer in text and never invoke tools

llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────
# 3. DEFINE THE AGENT NODE
# ─────────────────────────────────────────
# MessagesState is a built-in TypedDict with one field:
#   messages: Annotated[list[AnyMessage], add_messages]
# The add_messages annotation means new messages are APPENDED,
# not overwritten — this is important for conversation history.

def agent_node(state: MessagesState) -> dict:
    """The only node we write ourselves. Calls the LLM and returns its response."""
    print(f"\n[Agent] Thinking... ({len(state['messages'])} messages in state)")
    response = llm_with_tools.invoke(state["messages"])
    # Returning a dict with 'messages' triggers the add_messages appender
    return {"messages": [response]}


# ─────────────────────────────────────────
# 4. BUILD THE GRAPH
# ─────────────────────────────────────────

builder = StateGraph(MessagesState)

# Only two nodes:
#   "agent"  — our function above (calls the LLM)
#   "tools"  — ToolNode (pre-built, runs whatever tool the LLM requested)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))   # pass the same tools list here

# Entry point
builder.add_edge(START, "agent")

# Conditional edge FROM "agent":
#   tools_condition checks state["messages"][-1] for tool_calls
#   → if tool_calls exist:  route to "tools"
#   → if no tool_calls:     route to END
builder.add_conditional_edges(
    "agent",
    tools_condition    # pre-built routing function — no need to write your own
)

# After ToolNode runs, always go back to the agent
# (so the LLM can see the tool result and decide what to do next)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ─────────────────────────────────────────
# 5. RUN IT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Agent ready. Try: 'What is 25 multiplied by 4?'")
    print("Or: 'What's the weather in Hanoi?'")
    print("Or: 'What is 10 + 5, and what's the weather in Tokyo?'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        # HumanMessage wraps our plain text into the format LangChain expects
        from langchain_core.messages import HumanMessage
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

        # The final message in state is always the LLM's last response
        print(f"\nAgent: {result['messages'][-1].content}")