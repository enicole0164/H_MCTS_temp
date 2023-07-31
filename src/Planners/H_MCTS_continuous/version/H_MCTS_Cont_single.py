from copy import deepcopy

import math
import random
import sys
import numpy as np

from src.Env.Grid.Higher_Grids_HW import HighLevelGrids3, HighLevelGrids2
from src.Planners.H_MCTS_continuous.Node_Cont import H_Node_Cont
from src.Env.Grid.Cont_Grid import Continuous_Grid


class H_MCTS_Cont:
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=1,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        l1_action_cost=(-1) * 2,
        iter_Limit=10000,
        explorationConstant_h=1 / math.sqrt(1.5),  # 1 / math.sqrt(2)
        explorationConstant_l=1 / math.sqrt(1.5),  # 1 / math.sqrt(2)
        random_seed=25,
        num_barrier=15,
        gamma=1,
        alpha=0.05,
        constant_c=10,
        assigned_barrier=None,
        assigned_start_goal=None,
        cont_action_radius=1,
    ):
        self.searchLimit = iter_Limit
        self.limitType = "iter"
        self.iteration = 0

        self.gamma = gamma

        # Set alpha
        self.alpha = alpha
        self.constant_c = constant_c

        self.l1_rows, self.l1_cols = grid_setting[0], grid_setting[1]
        self.l1_width, self.l1_height = grid_setting[2], grid_setting[3]
        self.A_space = A_space
        self.RS = RS
        self.explorationConstant_h = explorationConstant_h
        self.explorationConstant_l = explorationConstant_l

        # set random
        np.random.seed(random_seed)
        

        self.set_env(
            grid_setting,
            H_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            l1_action_cost,
            random_seed,
            num_barrier,
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
        H_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        l1_action_cost=(-1) * 2,
        random_seed=25,
        num_barrier=10,
        assigned_barrier=None,
        assigned_start_goal=None,
        cont_action_radius=1,
    ):  
        self.cont_env = Continuous_Grid(
            grid_settings=grid_setting[:4],
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
            goal_n_start_distance=grid_setting[4],
            l1_goal_reward=l1_goal_reward,
            l1_subgoal_reward=l1_subgoal_reward,
            assigned_barrier=assigned_barrier,
            assigned_start_goal=assigned_start_goal,
            cont_action_radius=cont_action_radius,
        )
        
        self.env = HighLevelGrids2(
            grid_settings=grid_setting,
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
            l1_goal_reward=l1_goal_reward,
            l1_subgoal_reward=l1_subgoal_reward,
            assigned_barrier=assigned_barrier,
            assigned_start_goal=assigned_start_goal,
        )
        
        self.env.inherit_start_goal(self.cont_env.start_dict, self.cont_env.goal_dict)

        # Generate initial state at the highest level
        self.init_s = (
            self.env.H_level,
            self.env.start_dict[self.env.H_level][0],
            self.env.start_dict[self.env.H_level][1],
        )

    def set_Root(self):
        self.root = H_Node_Cont(self.init_s, deepcopy(self.env), parent=None)

    # while loop in pseudo code (replace by for loop)
    def search(self, tree_save_path=None, traj_save_path=None):
        success = False
        self.set_Root()  # set root and its subgoal ( destination at high level)

        if traj_save_path:
            stdout_fileno = sys.stdout
            # Redirect sys.stdout to the file
            sys.stdout = open(traj_save_path, 'w')

        for i in range(self.searchLimit):
            path = self.executeRound()
            if path is not None:
                success = True
                self.cont_env.plot_grid_tree(self.root, save_path=tree_save_path)
                if traj_save_path:
                    sys.stdout.close()
                    sys.stdout = stdout_fileno
                return path, success, i + 1

        self.cont_env.plot_grid_tree(self.root, save_path=tree_save_path)
        if traj_save_path:
            sys.stdout.close()
            sys.stdout = stdout_fileno
        return None, False, i + 1

    # One Simulation
    def executeRound(self):
        cur_root_level = self.root.s[0]

        # Select Leaf
        node = self.selectNode(self.root)
        # print("node state: ", node.s)
        
        node_start, subgoal_traj = self.Backtrace(node)

        # Found the path
        if node.isTerminal:
            print("FOUND PATH IN LEVEL ",subgoal_traj[0][0] , subgoal_traj)
            if len(self.success_traj[node.s[0]]) == 0:  # first path at level node.s[0]
                if cur_root_level > 0:  # FOUND high level path
                    self.Root_renew()  # ROOT level down
                    cur_root_level -= 1

                elif cur_root_level == 0:  # FOUND the route at level 1
                    return node.traj

            # Reset subgoal and delete subgoal route
            self.SetNewSubgoal(node_start=node_start, subgoal_traj=subgoal_traj)

            # Delete the whole tree and leave root node only
            self.delete_high_tree(node_start=node_start)

            self.success_traj[node.s[0]].add(tuple(node.traj))
            
        self.backpropagate(node=node, node_start=node_start)
        
    # Only operate When find the path of high-level
    def delete_child(self, node_start: H_Node_Cont, subgoal_traj: list):
        for action in node_start.children.keys():
                if node_start.children[action].s == subgoal_traj[0]:
                    del node_start.children[action]
                    break
    
    # Only operate When find the path of high-level
    def delete_high_tree(self, node_start: H_Node_Cont):
        node_start.children.clear()

    # SELECTLEAF in pseudo code
    def selectNode(self, node: H_Node_Cont):
        while not node.isTerminal:  # and not node.isCycle:
            if node.isFullyExpanded:
                if node.s[0] == 0:
                    assert(False) # never chooses BestChild for node.s[0] == 0
                node = self.getBestChild(node, self.explorationConstant_h)
            elif node.s[0] == 0:
                numVisits = node.numVisits
                expand_limit = round(self.constant_c * (numVisits**self.alpha))

                if len(node.children) >= expand_limit:
                    node = self.getBestChild(node, self.explorationConstant_l)
                # Exploration
                else:
                    return self.expand(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node: H_Node_Cont):
        
        if node.s[0] > 0:    
            action = random.choice(list(node.untried_Actions))
        else:
            action = node.getPossibleAction()
        next_s = node.step(action)

        if action[0] > 0:   
            new_Node = H_Node_Cont(s=next_s, env=deepcopy(self.env), parent=node)
            assert(next_s[0] == node.s[0])
            for child in node.children.values():
                if (child.s == next_s):
                    assert(False)
            node.untried_Actions.remove(action)
        else:
            new_Node = H_Node_Cont(s=next_s, env=deepcopy(self.cont_env), parent=node)

        
        node.children[action] = new_Node

        # Compare only ***current level***'s children
        if node.s[0] > 0:
            self.checkFullExpand(node)
        return new_Node

    def checkFullExpand(self, node: H_Node_Cont):
        if len(node.untried_Actions) == 0:
            node.isFullyExpanded = True

    def Root_renew(self):  # root to low level
        new_root_level = self.root.s[0] - 1
        root_state = (
            new_root_level,
            self.env.start_dict[new_root_level][0],
            self.env.start_dict[new_root_level][1],
        )
        self.root.s = root_state
        if new_root_level == 0:
            self.root.env = self.cont_env
            self.root.traj = [root_state]
            self.root.traj_dict[new_root_level] = [root_state[1:]]

        else:  # allow ACTION stay (0, 0)
            self.root.env = self.env

            self.root.untried_Actions = deepcopy(self.root.getPossibleActions())
            self.root.traj_dict[new_root_level] = []

        self.root.isFullyExpanded = False
        self.root.distance = self.root.get_distance()
        self.root.totalReward = 0.0
        self.root.numVisits = 0

    def backpropagate(self, node: H_Node_Cont, node_start: H_Node_Cont):
        reward = self.getReward(node)

        if node is node_start:
            node.numVisits += 1
            node.totalReward += reward
            return

        while True:  # root = parent is None
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
            if node is node_start:  # if node is self.root:
                node.numVisits += 1
                break

            reward = self.gamma * reward
            if node.s[0] != 0:
                reward += self.getReward(node)

    def getBestChild(self, node: H_Node_Cont, explorationValue: float):
        bestValue = float("-inf")
        bestNodes = []
        if not node.children.values():
            # print(node.traj)
            raise Exception(
                f"No CHILD AT {node.s, node.untried_Actions, node.achieved_subgoal, node.subgoal_set}"
            )

        for child in node.children.values():
            # calculate UCT value
            nodeValue = ( # calculate the best child
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:  # same UCT value
                bestNodes.append(child)
        return random.choice(bestNodes)

    # route from root to bestChild
    def getAction(self, parent: H_Node_Cont, child: H_Node_Cont):
        for action, node in parent.children.items():
            if node is child:
                return action

        raise ValueError("there is no relation between parent and child")

    def getReward(self, node: H_Node_Cont):
        if node.s[0] != 0:
            return self.env.calculate_reward(node=node)
        else:
            return self.cont_env.calculate_reward(node=node)

    # When find the high-level path
    def SetNewSubgoal(self, node_start: H_Node_Cont, subgoal_traj: list):
        node_start.subgoal_set.add(tuple(subgoal_traj))

    # Find the start node (Root or Extendable)
    def Backtrace(self, node: H_Node_Cont):  # not extendable extendable and high level.
        start_level = node.s[0]
        traj = []
        # while not (node.isRoot or node.isExtendable):
        while not (node.isRoot or (node.isExtendable and node.s[0] < start_level)): 
            traj.insert(0, node.s)  # leaf to extendable, leaf to root
            node = node.parent
        return node, traj
    
    def draw_tree(self, node: H_Node_Cont, depth=0):
        indent = "    " * depth  # Indentation for visualizing depth
        
        # Print node value and depth
        print(indent + "|-- " , node.s)
        print(indent + "|-- " , node.subgoal_set)
        print(indent + "|-- depth", depth)
        
        # Recursively draw children
        for index, child in enumerate(node.children.values()):
            if index == len(node.children) - 1:
                print(indent + "    ")
                self.draw_tree(child, depth+1)
            else:
                print(indent + "|")
                self.draw_tree(child, depth+1)
