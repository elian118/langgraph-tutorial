from langgraph.checkpoint.memory import MemorySaver

from utils import visualize_graph
from work_flow import workflow

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

if __name__ == '__main__':
    visualize_graph(app)
