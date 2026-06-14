import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
#from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Load biến môi trường
load_dotenv()

# 1. Khởi tạo LLM (Sử dụng GPU của bạn)
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
#llm = ChatGroq(model="", temperature=0)
# 2. Định nghĩa State (Trạng thái của Graph)
class State(TypedDict):
    question: str
    answer: str

# 3. Định nghĩa Node
def call_llm(state: State):
    print("---ĐANG GỌI GPU---")
    response = llm.invoke(state["question"])
    return {"answer": response.content}

# 4. Xây dựng Graph
workflow = StateGraph(State)
workflow.add_node("agent", call_llm)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Compile Graph
app = workflow.compile()

# 5. Chạy thử
if __name__ == "__main__":
    inputs = {"question": "Chào bạn"}
    for event in app.stream(inputs):
        for value in event.values():
            print("AI Trả lời:", value["answer"])