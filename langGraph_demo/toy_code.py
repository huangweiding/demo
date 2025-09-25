from __future__ import annotations
from typing import Dict, List, Any, TypedDict
from typing_extensions import Annotated
import operator

from langgraph.graph import StateGraph, END


class ModelClient:
    def __init__(self, model: str = "gpt-mock") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        return f"[{self.model}] echo: {prompt}"


class GraphState(TypedDict, total=False):
    user_id: str
    session_id: str
    flow_id: str
    status: str
    last_output: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    vars: Dict[str, Any]
    _flags: Dict[str, bool]


def node_prepare(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "准备中"}]
    return {
        "status": "prepare",
        "messages": messages,
    }


def node_llm_respond(state: GraphState, client: ModelClient) -> GraphState:
    prompt = state.get("vars", {}).get("query", "")
    fake_reply = f"LLM回复：收到你的问题[{prompt}]"
    messages = state.get("messages", []) + [{"role": "assistant", "content": fake_reply}]
    return {
        "status": "responding",
        "last_output": fake_reply,
        "messages": messages,
    }


def node_finalize(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "流程结束"}]
    return {
        "status": "done",
        "messages": messages,
    }

# --- 订单状态相关节点 ---

def node_pay(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：待支付"}]
    return {"messages": messages}


def node_ticketing(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：出票中"}]
    return {"messages": messages}


def node_partial_issue(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：部分出票"}]
    return {"messages": messages}


def node_canceled(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：已取消"}]
    return {"messages": messages}


def node_refund_all(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：全部退票"}]
    return {"messages": messages}


def node_refund_partial(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：部分退票"}]
    return {"messages": messages}


def node_cancel_partial(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：部分取消"}]
    return {"messages": messages}


def node_issued(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：已出票"}]
    return {"messages": messages}


def node_fallback(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "状态：未知/兜底"}]
    return {"messages": messages}

# --- “部分出票”二级分支节点 ---
def node_partial_outbound(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "二级：去程未出票"}]
    return {"messages": messages}

def node_partial_return(state: GraphState, client: ModelClient) -> GraphState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": "二级：返程未出票"}]
    return {"messages": messages}


def to_mermaid_pretty() -> str:
    lines: List[str] = ["flowchart TD"]
    # 起止
    lines.append("    START([start]) --> prepare")
    # 一级条件路由（与 main 中一致）
    order_routes: Dict[str, str] = {
        "待支付":   "node_pay",
        "出票中":   "node_ticketing",
        "部分出票": "node_partial_issue",
        "已取消":   "node_canceled",
        "全部退票": "node_refund_all",
        "部分退票": "node_refund_partial",
        "部分取消": "node_cancel_partial",
        "已出票":   "node_issued",
        "unknown":  "node_fallback",
    }
    for label, dst in order_routes.items():
        lines.append(f"    prepare -- {label} --> {dst}")
    # 二级条件：部分出票
    partial_routes: Dict[str, str] = {
        "outbound": "node_partial_outbound",
        "return":   "node_partial_return",
        "unknown":  "finalize",
    }
    for label, dst in partial_routes.items():
        lines.append(f"    node_partial_issue -- {label} --> {dst}")
    # 固定收尾
    for n in [
        "node_pay","node_ticketing","node_partial_issue","node_canceled",
        "node_refund_all","node_refund_partial","node_cancel_partial","node_issued",
        "node_fallback","node_partial_outbound","node_partial_return",
    ]:
        lines.append(f"    {n} --> finalize")
    lines.append("    finalize --> END([end])")
    return "\n".join(lines)


def main() -> None:
    client = ModelClient(model="gpt-mock")

    # 使用 LangGraph StateGraph
    graph = StateGraph(GraphState)

    # 注册节点（封装以便复用 client）
    graph.add_node("prepare", lambda s: node_prepare(s, client))
    graph.add_node("finalize", lambda s: node_finalize(s, client))
    graph.add_node("node_pay", lambda s: node_pay(s, client))
    graph.add_node("node_ticketing", lambda s: node_ticketing(s, client))
    graph.add_node("node_partial_issue", lambda s: node_partial_issue(s, client))
    graph.add_node("node_partial_outbound", lambda s: node_partial_outbound(s, client))
    graph.add_node("node_partial_return", lambda s: node_partial_return(s, client))
    graph.add_node("node_canceled", lambda s: node_canceled(s, client))
    graph.add_node("node_refund_all", lambda s: node_refund_all(s, client))
    graph.add_node("node_refund_partial", lambda s: node_refund_partial(s, client))
    graph.add_node("node_cancel_partial", lambda s: node_cancel_partial(s, client))
    graph.add_node("node_issued", lambda s: node_issued(s, client))
    graph.add_node("node_fallback", lambda s: node_fallback(s, client))

    # 入口
    graph.set_entry_point("prepare")

    # 条件化出边：按订单状态路由
    def router_order(s: GraphState) -> str:
        return s.get("vars", {}).get("order_status", "unknown")

    graph.add_conditional_edges(
        "prepare",
        router_order,
        {
            "待支付":   "node_pay",
            "出票中":   "node_ticketing",
            "部分出票": "node_partial_issue",
            "已取消":   "node_canceled",
            "全部退票": "node_refund_all",
            "部分退票": "node_refund_partial",
            "部分取消": "node_cancel_partial",
            "已出票":   "node_issued",
            "unknown":  "node_fallback",
        },
    )

    # 各状态节点统一流向 finalize -> END
    for n in [
        "node_pay","node_ticketing","node_partial_issue","node_canceled",
        "node_refund_all","node_refund_partial","node_cancel_partial","node_issued",
        "node_fallback",
    ]:
        graph.add_edge(n, "finalize")
    graph.add_edge("finalize", END)

    # 为“部分出票”增加二级条件路由
    def router_partial(s: GraphState) -> str:
        return s.get("vars", {}).get("partial_detail", "unknown")

    graph.add_conditional_edges(
        "node_partial_issue",
        router_partial,
        {
            "outbound": "node_partial_outbound",
            "return":   "node_partial_return",
            "unknown":  "finalize",
        },
    )
    graph.add_edge("node_partial_outbound", "finalize")
    graph.add_edge("node_partial_return", "finalize")

    # 编译
    app = graph.compile()

    # 初始状态
    state: GraphState = {
        "user_id": "u1",
        "session_id": "s1",
        "flow_id": "f1",
        "status": "init",
        "last_output": "",
        "messages": [],
        "vars": {},
        "_flags": {},
    }

    # 示例：先走“部分出票”，再根据 partial_detail 分流
    state["vars"]["order_status"] = "部分出票"
    state["vars"]["partial_detail"] = "outbound"  # 改成 "return" 或删除观察兜底

    final_state = app.invoke(state)

    print("status:", final_state.get("status"))
    print("last_output:", final_state.get("last_output"))
    print("messages:")
    for m in final_state.get("messages", []):
        print("-", m)

    # 打印简洁版 Mermaid 图
    print("\nMermaid diagram (pretty):\n")
    print(to_mermaid_pretty())


if __name__ == "__main__":
    main()
