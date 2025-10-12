from __future__ import annotations
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from tools import get_merchant_cls
import json
import traceback

class ReasonData(BaseModel):
    """분석 근거 데이터의 구조"""
    column: str = Field(description="분석에 사용된 데이터 컬럼명")
    value: Any = Field(description="해당 컬럼의 실제 값")
    context: str = Field(description="해당 값이 가지는 의미나 맥락에 대한 설명")

class Strategy(BaseModel):
    """전략 데이터 구조"""
    explanation: str = Field(
        description="전략에 대한 전반적인 설명과 데이터 기반의 근거"
    )
    example: str = Field(
        description="실행 가능한 구체적인 액션 플랜"
    )

class DataAnalysisOutput(BaseModel):
    """Data Analyzer 에이전트의 출력 구조"""
    store_info: List[str] = Field(description="[가맹점명, 업종, 상권] 정보")
    request: Optional[List[str]] = Field(description="사용자의 추가 요청사항 리스트")
    analysis: str = Field(description="분석 결과 요약")
    marketing_problem: str = Field(description="가장 시급하게 해결해야 할 핵심 마케팅 문제점")
    reason_data: List[ReasonData] = Field(description="분석의 핵심 근거가 되는 데이터 목록")

class GoalSetterOutput(BaseModel):
    """Goal Setter 에이전트의 출력 구조"""
    marketing_goal: str = Field(description="데이터 분석 기반으로 설정된 구체적인 마케팅 목표")
    goal_reason: str = Field(description="해당 목표를 설정한 핵심 이유")

class MarketerOutput(BaseModel):
    """Marketer 에이전트의 출력 구조"""
    reason: str = Field(description="해당 전략들을 제안하는 데이터 기반 이유.")
    strategy_1: Strategy = Field(description="첫 번째 핵심 마케팅 전략과 구체적인 실행 방안.")
    strategy_2: Strategy = Field(description="두 번째 핵심 마케팅 전략과 구체적인 실행 방안.")
    strategy_3: Strategy = Field(description="세 번째 핵심 마케팅 전략과 구체적인 실행 방안.")

class PipelineState(BaseModel):
    """LangGraph의 전체 상태를 관리하는 Pydantic 모델"""
    user_query: str
    data_analysis_result: Optional[DataAnalysisOutput] = None
    goal_setter_result: Optional[GoalSetterOutput] = None
    marketer_result: Optional[MarketerOutput] = None
        
    final_output: str = Field(default="")

BASE_DIR = Path(__file__).resolve().parent

with open(f"{BASE_DIR}/data/prompts/data_analyzer_prompt.txt", "r", encoding="utf-8") as fr:
    SYS_DATA_ANALYZER = fr.read()
with open(f"{BASE_DIR}/data/prompts/goal_setter_prompt.txt", "r", encoding="utf-8") as fr:
    SYS_GOAL_SETTER = fr.read()
with open(f"{BASE_DIR}/data/prompts/marketer_prompt.txt", "r", encoding="utf-8") as fr:
    SYS_MARKETER = fr.read()
with open(f"{BASE_DIR}/data/prompts/summarizer_prompt.txt", "r", encoding="utf-8") as fr:
    SYS_SUMMARIZER = fr.read()

def init_mcp_tools(tools, tool_config: Dict[str, List[str]]) -> Dict[str, List[Any]]:
    if not tools:
        raise ValueError("the tool set can not be loaded")
    def pick(prefixes):
        return [t for t in tools if any(t.name.startswith(p) for p in prefixes)]
    return {
        "data_analyzer": pick(tool_config.get("data_analyzer", [])),
        "goal_setter": pick(tool_config.get("goal_setter", [])),
        "marketer": pick(tool_config.get("marketer", [])),
        "summarizer": pick(tool_config.get("summarizer", []))
    }

def check_continue(state: PipelineState) -> str:
    """data_analyzer 실행 후 final_output에 값이 채워졌는지 (오류 발생 여부) 확인"""
    if state.final_output:
        return "end"
    else:
        return "continue"
    
async def call_tool(ai: AIMessage, tool_map):
    if ai.tool_calls:
        tool_messages = []
        for tool_call in ai.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            if tool_name in tool_map:
                function_to_call = tool_map[tool_name]
                tool_result = await function_to_call.ainvoke(tool_args)
                
                if not isinstance(tool_result, (dict, list, str, int, float, bool, type(None))):
                    serializable_result = tool_result.__dict__
                else:
                    serializable_result = tool_result
                
                tool_messages.append(ToolMessage(
                    content=json.dumps(serializable_result, ensure_ascii=False),
                    tool_call_id=tool_id
                ))  
        return tool_messages
    else:
        return []
    
async def data_analyzer(state: PipelineState, llm: ChatGoogleGenerativeAI, tools):
    sys = SystemMessage(content=SYS_DATA_ANALYZER)
    user = HumanMessage(content=state.user_query)
    st_llm = llm.with_structured_output(DataAnalysisOutput)
    if tools:
        tool_map = {f.name: f for f in tools}   # 함수 이름을 기준으로 map 생성
        bind_llm = llm.bind_tools(tools)
        
        ai: AIMessage = await bind_llm.ainvoke([sys, user])
        
        if not ai.tool_calls:
            return {
                "final_output": ai.content
            }
    
        tool_messages = await call_tool(ai, tool_map)
        messages = [sys, user, ai] + (tool_messages or [])
        response = await st_llm.ainvoke(messages)
        response.store_info[1] = get_merchant_cls(response.store_info[1])
    else:
        response = await st_llm.ainvoke([sys, user])

    return {"data_analysis_result": response.model_dump()}
    
async def goal_setter(state: PipelineState, llm: ChatGoogleGenerativeAI, tools):
    user_input = state.data_analysis_result.model_dump_json(include={'store_info', 'request', 'analysis', 'marketing_problem', 'reason_data'})
    sys = SystemMessage(content=SYS_GOAL_SETTER)
    user = HumanMessage(content=user_input)
    st_llm = llm.with_structured_output(GoalSetterOutput)

    if tools:
        tool_map = {f.name: f for f in tools}
        bind_llm = llm.bind_tools(tools)
        
        ai: AIMessage = await bind_llm.ainvoke([sys, user])
    
        tool_messages = await call_tool(ai, tool_map)
        if tool_messages:
            final_messages = [sys, user, ai] + tool_messages
            response = await st_llm.ainvoke(final_messages)
        else:
            response = await st_llm.ainvoke([sys, user, ai])
    else:
        response = await st_llm.ainvoke([sys, user])
    return {"goal_setter_result": response.model_dump()}


async def marketer(state: PipelineState, llm: ChatGoogleGenerativeAI, tools):
    user_input = {
        "store_info": state.data_analysis_result.store_info,
        "marketing_goal": state.goal_setter_result.marketing_goal,
        "analysis": state.data_analysis_result.analysis,
        "reason_data": [rd.model_dump() for rd in state.data_analysis_result.reason_data]
    }
    sys = SystemMessage(content=SYS_MARKETER)
    user = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))
    st_llm = llm.with_structured_output(MarketerOutput)
    
    if tools:
        tool_map = {f.name: f for f in tools}        
        bind_llm = llm.bind_tools(tools)
        
        ai: AIMessage = await bind_llm.ainvoke([sys, user])
    
    
        tool_messages = await call_tool(ai, tool_map)
        if tool_messages:
            final_messages = [sys, user, ai] + tool_messages
            response = await st_llm.ainvoke(final_messages)
        else:
            response = await st_llm.ainvoke([sys, user, ai])
    else:
        response = await st_llm.ainvoke([sys, user])
    return {"marketer_result": response.model_dump()}

async def summarizer(state: PipelineState, llm: ChatGoogleGenerativeAI, tools):
    user_input = {
        "store_info": state.data_analysis_result.store_info,
        "marketing_problem": state.data_analysis_result.marketing_problem,
        "marketing_goal": state.goal_setter_result.marketing_goal,
        "strategy_1": state.marketer_result.strategy_1.model_dump(),
        "strategy_2": state.marketer_result.strategy_2.model_dump(),
        "strategy_3": state.marketer_result.strategy_3.model_dump(),
        "reason": state.marketer_result.reason,
        "reason_data": [rd.model_dump() for rd in state.data_analysis_result.reason_data],
    }   
    sys = SystemMessage(content=SYS_SUMMARIZER)
    user = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))
    
    if tools:
        tool_map = {f.name: f for f in tools}
        bind_llm = llm.bind_tools(tools)
        
        ai: AIMessage = await bind_llm.ainvoke([sys, user])
    
    
        tool_messages = await call_tool(ai, tool_map)
        if tool_messages:
            final_messages = [sys, user, ai] + tool_messages
            response = await llm.ainvoke(final_messages)
        else:
            response = await llm.ainvoke([sys, user, ai])
    else:
        response = await llm.ainvoke([sys, user])
    return {"final_output": response.content}    

def build_graph(llms: Dict[str, ChatGoogleGenerativeAI], tools: Dict[str, List[Any]]):
    data_analyzer_tools = tools.get('data_analyzer', [])
    goal_setter_tools = tools.get('goal_setter', [])
    marketer_tools = tools.get('marketer', [])
    summarizer_tools = tools.get('summarizer', [])
    
    graph = StateGraph(PipelineState)
    
    graph.add_node("data_analyzer", partial(data_analyzer, llm=llms['data_analyzer'], tools=data_analyzer_tools))    
    graph.add_node("goal_setter", partial(goal_setter, llm=llms['goal_setter'], tools=goal_setter_tools))
    graph.add_node("marketer", partial(marketer, llm=llms['marketer'], tools=marketer_tools))    
    graph.add_node("summarizer", partial(summarizer, llm=llms['summarizer'], tools=summarizer_tools))

    graph.add_edge(START, "data_analyzer")
    graph.add_conditional_edges(
        "data_analyzer",
        check_continue,
        {
            "continue": "goal_setter",
            "end": END
        }
    )
    graph.add_edge("goal_setter", "marketer")
    graph.add_edge("marketer", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile(checkpointer=MemorySaver())