from graph_state import GraphState

def retrieve(state: GraphState) -> GraphState:
    document = "검색된 문서"
    return GraphState(context=document)

def rewrite_query(state: GraphState) -> GraphState:
    document = "검색된 문서"
    return GraphState(context=document)

def llm_gpt_execute(state: GraphState) -> GraphState:
    answer = "GPT 생성된 답변"
    return GraphState(answer=answer)

def llm_claude_execute(state: GraphState) -> GraphState:
    answer = "Claude 생성된 답변"
    return GraphState(answer=answer)

def relevance_check(state: GraphState) -> GraphState:
    binary_score = "Relevance Score"
    return GraphState(binary_score=binary_score)

def sum_up(state: GraphState) -> GraphState:
    # sum_up: 결과 종합
    answer = "종합된 답변"
    return GraphState(answer=answer)

def search_on_web(state: GraphState) -> GraphState:
    # sum_up: 웹 검색
    document = state["context"] = "기존 문서"
    searched_documents = "검색된 문서"
    document += searched_documents
    return GraphState(context=document)

def get_table_info(state: GraphState) -> GraphState:
    table_info = "테이블 정보"
    return GraphState(context=table_info)

def generate_sql_query(state: GraphState) -> GraphState:
    sql_query = "SQL 쿼리"
    return GraphState(sql_query=sql_query)

def execute_sql_query(state: GraphState) -> GraphState:
    sql_result = "SQL 결과"
    return GraphState(context=sql_result)

def validate_sql_query(state: GraphState) -> GraphState:
    binary_score = "SQL 쿼리 검증 결과"
    return GraphState(binary_score=binary_score)

def handle_error(state: GraphState) -> GraphState:
    error = "에러 발생"
    return GraphState(context=error)

def decision(state: GraphState) -> GraphState:
    decision = "결정"

    if state["binary_score"] == "yes":
        return "종료"
    else:
        return "재검색"

    # return GraphState(context=decision)
