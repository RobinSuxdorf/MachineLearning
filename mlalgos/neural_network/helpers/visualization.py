from graphviz import Digraph
from mlalgos.neural_network import Value

def _trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """
    Trace the computational graph starting from the root Value node.

    This function performs a depth-first traversal of the computational graph 
    starting from the given root node and collects all nodes and edges involved 
    in the computation. 

    Args:
        root (Value): The root node of the computational graph from which to start tracing.

    Returns:
        tuple[set[Value], set[tuple[Value, Value]]]: A tuple containing:
            - A set of all Value nodes encountered in the computational graph.
            - A set of edges represented as tuples of parent-child relationships between Value nodes.
    """
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
    """
    Draw the computational graph using Graphviz, starting from the root Value node.

    This function visualizes the computational graph by creating a directed acyclic graph (DAG) 
    using Graphviz. It represents each node's data and gradient and draws edges for operations 
    performed between nodes.

    Args:
        root (Value): The root node of the computational graph to visualize.

    Returns:
        Digraph: A Graphviz Digraph object representing the computational graph.
    """
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = _trace(root)
    for n in nodes:
        uid = str(id(n))
        graph.node(name = uid, label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape="record")

        if n._op:
            graph.node(name = uid + n._op, label = n._op)
            graph.edge(uid + n._op, uid)

    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2._op)

    return graph
