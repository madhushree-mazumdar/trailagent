from graphviz import Digraph

def create_trailagent_workflow_dag(filename="trailagent_workflow_dag.png"):
    dot = Digraph(comment="TrailAgent LangGraph Workflow", format="png")
    #dot.attr(rankdir="LR", size="8,8")
    # Global Graph Attributes
    dot.attr(
        bgcolor='#222222',
        fontname='Georgia',
        fontsize='12',
        fontcolor='#DDDDDD',
        nodesep='0.5',
        ratio='0.3',
        ranksep='0.7',
        rankdir='LR',      # Left to Right
        splines='polyline'   # Smooth, modern lines
    )

    dot.attr('node',
        shape='rect',
        style='filled, rounded',
        fillcolor='#444444',
        color='#DDDDDD',      # Light grey border
        fontname='Georgia',
        fontcolor='#DDDDDD',
        margin='0.2'
    )

    dot.attr('edge',
        color='#AAAAAA',      # Dark grey instead of black
        arrowsize='0.8',      # Slightly smaller arrows
        fontname='Georgia',
        fontsize='10',
        fontcolor='#DDDDDD'
    )

    # Nodes
    dot.node("input_guard", "Input Guardrail")
    dot.node("retrieve", "Retrieve Context")
    dot.node("generate", "Generate Answer")
    dot.node("output_guard", "Output Guardrail")
    dot.node("halt_process", "Halt Process", shape="box", style="filled", color="red")
    dot.node("END", "END", shape="doublecircle", style="filled", color="lightgrey")

    # Edges for input_guard
    dot.edge("input_guard", "retrieve", label="SAFE")
    dot.edge("input_guard", "halt_process", label="UNSAFE_INPUT")

    # Main RAG flow
    dot.edge("retrieve", "generate")
    dot.edge("generate", "output_guard")

    # Edges for output_guard
    dot.edge("output_guard", "END", label="SAFE")
    dot.edge("output_guard", "halt_process", label="UNSAFE_OUTPUT")

    # Halt process always goes to END
    dot.edge("halt_process", "END")

    dot.render(filename, view=False)
    print(f"Workflow DAG image saved as {filename}")

if __name__ == "__main__":
    create_trailagent_workflow_dag()