from langgraph.graph import StateGraph, END

from intro.graph_state import GraphState
from intro.nodes import retrieve, relevance_check, llm_gpt_execute, sum_up, decision, llm_claude_execute, rewrite_query

# (1): Conventional RAG
# (2): 재검색
# (3): 멀티 LLM
# (4): 쿼리 재작성

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite_query", rewrite_query) # (4)
workflow.add_node("GPT 요청", llm_gpt_execute)
workflow.add_node("Claude 요청", llm_claude_execute) # (3)
workflow.add_node("GPT_relevance_check", relevance_check)
workflow.add_node("Claude_relevance_check", relevance_check) # (3)
workflow.add_node("결과 종합", sum_up)

workflow.add_edge("retrieve", "GPT 요청")
workflow.add_edge("retrieve", "Claude 요청") # (3)
workflow.add_edge("rewrite_query", "retrieve") # (4)
workflow.add_edge("GPT 요청", "GPT_relevance_check")
workflow.add_edge("GPT_relevance_check", "결과 종합")
workflow.add_edge("Claude 요청", "Claude_relevance_check") # (3)
workflow.add_edge("Claude_relevance_check", "결과 종합") # (3)

# workflow.add_edge("결과 종합", END) # (2)

# 조건부 엣지 추가. (2), (4)
# workflow.add_conditional_edges(
#     "결과 종합", decision, {
#         "재검색": "retrieve",
#         "종료": END,
#     }
# )

# 조건부 엣지 추가. (4)
workflow.add_conditional_edges(
    "결과 종합", decision, {
        "재검색": "rewrite_query",
        "종료": END,
    }
)

workflow.set_entry_point("retrieve")