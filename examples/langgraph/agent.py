from typing import List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, START, END

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

class GraphState(TypedDict):
    question: str
    documents: List[str]
    messages: List[str]


def explain(state: GraphState):
    prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
    Your job is to answer the provided question so that even a 5 year old can understand it. 
    You have provided with relevant background context to answer the question.

    Question: {question} 

    Answer:"""

    question = state["question"]
    documents = state.get("documents", [])
    formatted = prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}

graph = StateGraph(GraphState)
graph.add_node("explain", explain)

graph.add_edge(START, "explain")
graph.add_edge("explain", END)

app = graph.compile()