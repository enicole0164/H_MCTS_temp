import numpy as np

import random
import math
import graphviz


from copy import deepcopy


from src.Env.Grid.Cont_Grid import Continuous_Grid


class Plain_Node_cont:
    def __init__(self, s: tuple, env: Continuous_Grid, parent=None):
        self.s = s  # state: (level, x, y)
        self.env = env
        self.H_level = self.env.H_level
        
        self.parent = parent
        self.parent_level()
        
        self.children = dict()  # key: action, value: children
        
        self.set_traj()

        self.numVisits = 0
        self.totalReward = 0.0
        self.achieved_subgoal = []

        # self.untried_Actions = deepcopy(self.getPossibleActions())

        self.set_R_status()  # set self.isRoot
        self.set_T_status()  # set self.isTerminal
        self.get_distance()

        # NOT terminal -> CAN have children
        # terminal -> CANNOT have children
        self.isFullyExpanded = self.isTerminal
        
    def set_traj(self):
        if self.parent is None:  # Root
            if self.s[0] == 0:   # Root at level 0
                self.traj = [self.s]
            
            else:
                self.traj = []
            
        else:  # non-Root
            self.traj = deepcopy(self.parent.traj)
            self.traj.append(self.s)

    def parent_level(self):
        if self.parent is not None:  # Non-Root
            if self.parent.s[0] > self.s[0]:  # parent level is bigger then child level
                raise Exception('wrong parent level')
                
    # set Root status
    def set_R_status(self):
        if self.parent is None:
            self.isRoot = True
        else:
            self.isRoot = False

    # Set the node belongs to terminal or not, level regardless
    def set_T_status(self):
        if self.isRoot == True:  # Root CANNOT be terminal node
            self.isTerminal = False
        else:  # level regardless
            self.isTerminal = self.env.check_termination(self.s)

    def get_distance(self):  # at its level, v_{approx}
        self.distance = self.env.calculate_d2Goal(s=self.s)

    def num_child(self):
        return len(self.children)

    def step(self, action):  # -> state
        return self.env.step(self.s, action)

    # Do not consider cycle
    def getPossibleAction(self):
        return self.env.get_possible_Action(self.s)
            
    def set_traj_dict(self):  # for high level trajectories
        for level, s in self.level_pos.items():
            if self.parent.isRoot:  # allow STAY
                self.traj_dict[level].append(s)
            else:  # 
                if self.traj_dict[level][-1] != s:  # check the last element of the list
                    self.traj_dict[level].append(s)
                    

class Plain_MCTS_Cont:
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=0,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        l1_action_cost=(-1) * 2,
        iter_Limit=10000,
        explorationConstant=1 / math.sqrt(2),  # 1 / math.sqrt(2)
        alpha=0.05,
        constant_c=10,
        random_seed=25,
        num_barrier=15,
        gamma=1,
        assigned_barrier=None,
        assigned_start_goal=None,
        cont_action_radius=1,
        stepbystep=False,
    ):
        self.searchLimit = iter_Limit
        self.limitType = "iter"
        self.iteration = 0

        self.gamma = gamma

        # constant for PW
        self.alpha = alpha
        self.constant_c = constant_c

        self.l1_rows, self.l1_cols = grid_setting[0], grid_setting[1]
        self.l1_width, self.l1_height = grid_setting[2], grid_setting[3]
        self.A_space = A_space
        self.RS = RS
        self.explorationConstant = explorationConstant

        self.stepbystep = stepbystep        

        self.set_env(
            grid_setting=grid_setting,
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
            goal_n_start_distance=grid_setting[4],
            assigned_barrier=assigned_barrier,
            assigned_start_goal=assigned_start_goal,
            cont_action_radius=cont_action_radius,
        )

        # Assume that we know the env
        self.success_traj = {level: set() for level in self.env.levels}

    # Set environment for Root node and children nodes
    # Have difference at action space
    def set_env(
        self,
        grid_setting,  # l1_rows, l1_cols, lf1_width, l1_height
        H_level=0,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        l1_action_cost=(-1) * 2,
        random_seed=25,
        num_barrier=10,
        goal_n_start_distance=3,
        assigned_barrier=None,
        assigned_start_goal=None,
        cont_action_radius=1,
    ):
        self.env = Continuous_Grid(
            grid_settings=grid_setting,
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
            goal_n_start_distance=goal_n_start_distance,
            assigned_barrier=assigned_barrier,
            assigned_start_goal=assigned_start_goal,
            cont_action_radius=cont_action_radius,
        )

        # Generate initial state at the highest level
        self.init_s = (
            self.env.H_level,
            self.env.start_dict[self.env.H_level][0],
            self.env.start_dict[self.env.H_level][1],
        )

    def set_Root(self):
        self.root = Plain_Node_cont(self.init_s, deepcopy(self.env), parent=None)

    # while loop in pseudo code (replace by for loop)
    def search(self, save_path=None):
        success = False
        self.set_Root()  # set root and its subgoal ( destination at high level)

        for i in range(self.searchLimit):
            path, succeed = self.executeRound()
            if self.stepbystep and i%100==1:
                input("Press Enter to continue...")
                self.visualize_tree(self.root)

            if succeed:
                success = True
                self.env.plot_grid_tree(self.root, save_path=save_path)
                return path, success, i + 1

        self.env.plot_grid_tree(self.root, save_path=save_path)
        return None, False, i + 1

    # One Simulation
    def executeRound(self):
        # Select Leaf
        node = self.selectNode(self.root)

        # Found the path
        if node.isTerminal:
            return node.traj, True

        # self.backpropagate(node=node, node_start=self.root)
        self.backpropagate(node=node)
        return node.traj, False

    # SELECTLEAF in pseudo code
    def selectNode(self, node: Plain_Node_cont):
        while not node.isTerminal:
            numVisits = node.numVisits
            if numVisits == 0:
                expand_limit = self.constant_c
            else:
                expand_limit = round(self.constant_c * (numVisits**self.alpha))
            
            # w = random.uniform(0, 1)
            # if len(node.children) == expand_limit or (w > 0.05 and node.children):

            # Expand until expand_limit then choose best child.
            if len(node.children) >= expand_limit:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)

        return node

    def expand(self, node: Plain_Node_cont):
        action = node.getPossibleAction()
        next_s = node.step(action)

        new_Node = Plain_Node_cont(s=next_s, env=deepcopy(self.env), parent=node)
        node.children[action] = new_Node

        return new_Node

    def backpropagate(self, node: Plain_Node_cont):
        reward = self.getReward(node)

        if node is self.root:
            node.numVisits += 1
            node.totalReward += reward
            return

        while True:  # root = parent is None
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
            if node is self.root:  # if node is self.root:
                node.numVisits += 1
                node.totalReward += reward
                break

            reward = self.gamma * reward

    def getBestChild(self, node: Plain_Node_cont, explorationValue: float):
        bestValue = float("-inf")
        bestNodes = []
        if not node.children.values():
            print(node.traj)
            raise Exception(
                f"No CHILD AT {node.s, node.untried_Actions, node.achieved_subgoal, node.subgoal_set}"
            )

        for child in node.children.values():
            # calculate UCT value
            nodeValue = (
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(math.log(node.numVisits) / child.numVisits)
            )
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:  # same UCT value
                bestNodes.append(child)
        return random.choice(bestNodes)

    # route from root to bestChild
    def getAction(self, parent: Plain_Node_cont, child: Plain_Node_cont):
        for action, node in parent.children.items():
            if node is child:
                return action

        raise ValueError("there is no relation between parent and child")

    def getReward(self, node: Plain_Node_cont):
        return self.env.calculate_reward(node=node)
    

    def visualize_tree(self, root, depth=0, save_path=None):
        dot = graphviz.Digraph()
        root_id = str(root.s)+"@"+str(depth)+"root"
        root_name = str(root.s)
        dot.node(root_id, root_name)

        def add_nodes_edges(node, depth):
            if depth == 7:
                return
            for child in node.children.values():
                # node is parent of child
                # Create child node
                child_id = str(child.s)+"@"+str(depth+1)+"parent"+str(node.s)
                child_name = str(child.s)
                child_UCT = (
                        child.totalReward / child.numVisits
                        + self.explorationConstant
                        * math.sqrt(math.log(node.numVisits) / child.numVisits)
                    )
                dot.node(child_id, child_name + "\n" + str(child_UCT))
                # if node is Root
                if node.isRoot:
                    parent_id = str(node.s)+"@"+str(depth)+"root"
                    dot.edge(parent_id, child_id)
                else:
                    parent_id = str(node.s)+"@"+str(depth)+"parent"+str(node.parent.s)
                    dot.edge(parent_id, child_id)
                add_nodes_edges(child, depth+1)

        add_nodes_edges(root, 0)
        dot.render(view=True, format='png')