from __future__ import annotations
from typing import Callable, Dict, List, Any


class GraphState:
    def __init__(self, user_id: str, session_id: str, flow_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.flow_id = flow_id
        self.status: str = "init"
        self.last_output: str = ""
        self.messages: List[Dict[str, Any]] = []
        self.vars: Dict[str, Any] = {}
        self._flags: Dict[str, bool] = {}

    def set_status(self, status: str) -> None:
        self.status = status

    def set_flag(self, name: str, value: bool) -> None:
        self._flags[name] = value

    def get_flag(self, name: str, default: bool = False) -> bool:
        return self._flags.get(name, default)


class ModelClient:
    def __init__(self, model: str = "gpt-mock") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        return f"[{self.model}] echo: {prompt}"


NodeFunc = Callable[[GraphState, ModelClient], GraphState]


class LangGraphLike:
    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFunc] = {}
        self._edges: List[str] = []  # 线性模式
        self._guards: Dict[str, Callable[[GraphState], bool]] = {}
        # 条件路由模式
        self._entry: str | None = None
        self._finish: set[str] = set()
        self._next_edges: Dict[str, str] = {}  # 固定边: src -> dst
        self._cond_routes: Dict[str, Dict[str, str]] = {}  # node -> {label -> next}
        self._routers: Dict[str, Callable[[GraphState], str]] = {}  # node -> router(state)->label

    def add_node(self, name: str, fn: NodeFunc) -> None:
        self._nodes[name] = fn

    def set_edges(self, order: List[str]) -> None:
        self._edges = order

    def set_guard(self, name: str, guard: Callable[[GraphState], bool]) -> None:
        self._guards[name] = guard

    # --- 条件化出边 API（类似 add_conditional_edges） ---
    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def set_finish_point(self, name: str) -> None:
        self._finish.add(name)

    def add_edge(self, src: str, dst: str) -> None:
        self._next_edges[src] = dst

    def add_conditional_edges(
        self,
        name: str,
        routes: Dict[str, str],
        router: Callable[[GraphState], str],
    ) -> None:
        # routes: {label -> next_node}
        self._cond_routes[name] = routes
        self._routers[name] = router

    def run(self, state: GraphState, client: ModelClient) -> GraphState:
        # 若配置了条件路由，则优先走“条件模式”；否则走线性模式
        if self._entry is not None:
            if self._entry not in self._nodes:
                raise ValueError("entry point not found: " + str(self._entry))
            curr = self._entry
            while True:
                # guard（可选）
                guard = self._guards.get(curr)
                if guard is None or guard(state):
                    state = self._nodes[curr](state, client)
                # 终止条件
                if curr in self._finish:
                    return state
                # 条件出边优先
                if curr in self._cond_routes:
                    router = self._routers[curr]
                    label = router(state)
                    table = self._cond_routes[curr]
                    if label not in table:
                        raise ValueError(f"no route for label '{label}' from node '{curr}'")
                    curr = table[label]
                    continue
                # 固定出边其次
                if curr in self._next_edges:
                    curr = self._next_edges[curr]
                    continue
                # 无出边则结束
                return state
        else:
            # 线性模式（与原实现一致）
            for name in self._edges:
                guard = self._guards.get(name)
                if guard is not None and not guard(state):
                    continue
                node = self._nodes[name]
                state = node(state, client)
            return state

    # --- Mermaid 导出 ---
    def to_mermaid(self) -> str:
        lines: List[str] = ["flowchart TD"]
        # 标记起止
        if self._entry is not None:
            lines.append(f"    START([start]) --> {self._entry}")
        for f in self._finish:
            lines.append(f"    {f} --> END([end])")
        # 固定边
        for src, dst in self._next_edges.items():
            lines.append(f"    {src} --> {dst}")
        # 条件边（标注标签）
        for src, table in self._cond_routes.items():
            for label, dst in table.items():
                lines.append(f"    {src} -- {label} --> {dst}")
        # 线性模式（若无条件模式时）
        if self._entry is None and self._edges:
            for a, b in zip(self._edges, self._edges[1:]):
                lines.append(f"    {a} --> {b}")
        return "\n".join(lines)


def node_prepare(state: GraphState, client: ModelClient) -> GraphState:
    state.set_status("prepare")
    state.messages.append({"role": "assistant", "content": "准备中"})
    return state


def node_llm_respond(state: GraphState, client: ModelClient) -> GraphState:
    state.set_status("responding")
    prompt = state.vars.get("query", "")
    fake_reply = f"LLM回复：收到你的问题[{prompt}]"
    state.last_output = fake_reply
    state.messages.append({"role": "assistant", "content": fake_reply})
    return state


def node_finalize(state: GraphState, client: ModelClient) -> GraphState:
    state.set_status("done")
    state.messages.append({"role": "assistant", "content": "流程结束"})
    return state

# --- 订单状态相关节点 ---

def node_pay(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：待支付"})
    return state


def node_ticketing(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：出票中"})
    return state


def node_partial_issue(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：部分出票"})
    return state


def node_canceled(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：已取消"})
    return state


def node_refund_all(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：全部退票"})
    return state


def node_refund_partial(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：部分退票"})
    return state


def node_cancel_partial(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：部分取消"})
    return state


def node_issued(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：已出票"})
    return state


def node_fallback(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "状态：未知/兜底"})
    return state

# --- “部分出票”二级分支节点 ---
def node_partial_outbound(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "二级：去程未出票"})
    return state

def node_partial_return(state: GraphState, client: ModelClient) -> GraphState:
    state.messages.append({"role": "assistant", "content": "二级：返程未出票"})
    return state


def main() -> None:
    graph = LangGraphLike()
    graph.add_node("prepare", node_prepare)
    graph.add_node("finalize", node_finalize)
    # 注册订单状态节点
    graph.add_node("node_pay", node_pay)
    graph.add_node("node_ticketing", node_ticketing)
    graph.add_node("node_partial_issue", node_partial_issue)
    graph.add_node("node_partial_outbound", node_partial_outbound)
    graph.add_node("node_partial_return", node_partial_return)
    graph.add_node("node_canceled", node_canceled)
    graph.add_node("node_refund_all", node_refund_all)
    graph.add_node("node_refund_partial", node_refund_partial)
    graph.add_node("node_cancel_partial", node_cancel_partial)
    graph.add_node("node_issued", node_issued)
    graph.add_node("node_fallback", node_fallback)

    # 条件化出边：按订单状态路由
    graph.set_entry_point("prepare")
    graph.set_finish_point("finalize")

    def router_order(s: GraphState) -> str:
        return s.vars.get("order_status", "unknown")

    graph.add_conditional_edges(
        "prepare",
        routes={
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
        router=router_order,
    )

    # 各状态节点统一流向 finalize
    for n in [
        "node_pay","node_ticketing","node_partial_issue","node_canceled",
        "node_refund_all","node_refund_partial","node_cancel_partial","node_issued",
        "node_fallback",
    ]:
        self_node = n  # 占位，保持风格一致
        graph.add_edge(self_node, "finalize")

    # 为“部分出票”增加二级条件路由
    def router_partial(s: GraphState) -> str:
        # partial_detail: "outbound" | "return" | 其他
        return s.vars.get("partial_detail", "unknown")

    graph.add_conditional_edges(
        "node_partial_issue",
        routes={
            "outbound": "node_partial_outbound",
            "return":   "node_partial_return",
            "unknown":  "finalize",  # 兜底直接收尾
        },
        router=router_partial,
    )
    # 二级节点统一流向 finalize
    graph.add_edge("node_partial_outbound", "finalize")
    graph.add_edge("node_partial_return", "finalize")

    state = GraphState(user_id="u1", session_id="s1", flow_id="f1")
    # 示例：先走“部分出票”，再根据 partial_detail 分流
    state.vars["order_status"] = "部分出票"
    state.vars["partial_detail"] = "outbound"  # 改成 "return" 或删除观察兜底
    client = ModelClient(model="gpt-mock")

    final_state = graph.run(state, client)

    print("status:", final_state.status)
    print("last_output:", final_state.last_output)
    print("messages:")
    for m in final_state.messages:
        print("-", m)

    # 打印 mermaid 图
    print("\nMermaid diagram:\n")
    print(graph.to_mermaid())


if __name__ == "__main__":
    main()
