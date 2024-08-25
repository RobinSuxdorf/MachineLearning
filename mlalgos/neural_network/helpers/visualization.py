from graphviz import Digraph
from mlalgos.neural_network import Value

def _trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def _build(v: Value) -> None:
        nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            _build(child)

    _build(root)
    return nodes, edges

def draw_computational_graph(root: Value) -> Digraph:
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = _trace(root)
    for n in nodes:
        uid = str(id(n))
        graph.node(name = uid, label = "{ data %.4f | grad %.4f }" % (n._data, n.grad), shape="record")

        if n._op:
            graph.node(name = uid + n._op, label = n._op)
            graph.edge(uid + n._op, uid)

    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2._op)

    return graph
