import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# api_key = os.getenv("ANTHROPIC_API_KEY") # 사용하기 전에 5달러 결재라도 할 것... ㅡ.,ㅡ
api_key = os.getenv("GOOGLE_API_KEY")

# Step 1: tools & model 정의
from langchain.tools import tool

# model = ChatAnthropic(
#     model="claude-sonnet-4-5-20250929",
#     api_key=api_key,
#     temperature=0
# )

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0
)



# tools 정의
@tool
def multiply(a: int, b: int) -> int:
    """a`와 `b` 곱하기.

    Args:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """`a`와 `b` 더하기.

    Args:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """`a`와 `b` 나누기.

    Args:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a / b


# tools로 LLM 증강하기
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langgraph.graph import add_messages
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task


# Step 2: model node 정의

@task
def call_llm(messages: list[BaseMessage]):
    """LLM이 도구를 호출할지 여부를 결정합니다"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="당신은 입력 집합에 대해 산술을 수행하는 데 도움이 되는 에이전트입니다."
            )
        ]
        + messages
    )


# Step 3: tool node 정의

@task
def call_tool(tool_call: ToolCall):
    """도구 호출을 수행합니다"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


# Step 4: 에이전트 정의

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        # tools 실행
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages

# Invoke: 호출
messages = [HumanMessage(content="3과 4를 더하면?")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")