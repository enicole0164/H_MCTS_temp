import matplotlib.pyplot as plt
import numpy as np
import os

# Dir path
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'experiments')
PLAIN_MCTS_EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR, 'Plain-MCTS-wandb')
H_MCTS_EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR, 'H-MCTS')
H_MCTS_SINGLE_EXPERIMENT_DIR = os.path.join(H_MCTS_EXPERIMENT_DIR, 'single-plan-wandb')
H_MCTS_MULTI_EXPERIMENT_DIR = os.path.join(H_MCTS_EXPERIMENT_DIR, 'multi-plan-wandb')
H_MCTS_EXTENDABLE_VER3_EXPERIMENT_DIR = os.path.join(H_MCTS_EXPERIMENT_DIR, 'extendable-plan-wTree-ver3')

# Helper Function
# Generates iteration vs success rate plot
def cumul_plot(iter_Limit, mcts_result, path):
    iteration_plot = np.zeros(iter_Limit+1)
    for iter in mcts_result["iter_cnt"].values():
        iteration_plot[iter-1]+=1

    def cumulative_count(iterations):
        x = np.arange(1, len(iterations))
        y = np.cumsum(np.array(iterations)[:-1])
        
        return x, y

    x, y = cumulative_count(iteration_plot)

    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Iteration')

    plt.xlim(1, iter_Limit)  # Set the x-axis limits
    plt.xticks(range(0, iter_Limit+1, 1000))  # Set the x-axis tick marks

    plt.savefig("{}/success_rate.png".format(path))
    plt.close()
    return x, y

# Helper Function
# Check and Make the directory if it doesn't exist
def make_param_dir(path):
    # check whether directory already exists
    if not os.path.exists(path):
        os.mkdir(path)
        print("Folder %s created!" % path)
        return False
    else:
        print("Folder %s already exists" % path)
        return True
