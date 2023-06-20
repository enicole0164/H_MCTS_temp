from copy import deepcopy

import math
import random

from src.Env.Grid.Higher_Grids_HW import HighLevelGrids
from src.Planners.Node_HW import H_Node_HW


class H_MCTS_HW:  # HW: HeeSang
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=1,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=(-1) * 2,
        cycle_penalty=(-1) * 0,  # NOT fixed yet.
        iter_Limit=10000,
        explorationConstant=1 / math.sqrt(2),  # 1 / math.sqrt(2)
        random_seed=25,
        num_barrier=15,
        gamma=1
    ):
        self.searchLimit = iter_Limit
        self.limitType = "iter"
        self.iteration = 0
        
        self.gamma = gamma

        self.l1_rows, self.l1_cols = grid_setting[0], grid_setting[1]
        self.l1_width, self.l1_height = grid_setting[2], grid_setting[3]
        self.A_space = A_space
        self.RS = RS

        self.explorationConstant = explorationConstant

        self.set_env(
            grid_setting,
            H_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
            cycle_penalty,
            random_seed,
            num_barrier,
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
        action_cost=(-1) * 2,
        cycle_penalty=(-1) * 100,
        random_seed=25,
        num_barrier=10,
    ):
        self.env = HighLevelGrids(
            grid_setting,
            H_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
            cycle_penalty,
            random_seed,
            num_barrier,
        )

        # Generate initial state at the highest level
        self.init_s = (
            self.env.H_level,
            self.env.start_dict[self.env.H_level][0],
            self.env.start_dict[self.env.H_level][1],
        )

    def set_Root(self):
        self.root = H_Node_HW(self.init_s, deepcopy(self.env), parent=None)

    # while loop in pseudo code (replace by for loop)
    def search(self):
        success = False
        self.set_Root()  # set root and its subgoal ( destination at high level)

        for i in range(self.searchLimit):
            path = self.executeRound()
            if path is not None:
                success = True
                break

        return path, success, i + 1

    # One Simulation
    def executeRound(self):
        cur_root_level = self.root.s[0]

        # Select Leaf
        node = self.selectNode(self.root)
        node_start, subgoal_traj = self.Backtrace(node)

        # Delete if Cycle
        if node.isCycle:
            self.delete_cycle(node)
            return
        
        # subgoal check

        if node.isTerminal:
            # Root go low level and set subgoal
            if len(self.success_traj[node.s[0]]) == 0:  # first path at level node.s[0]
                if cur_root_level > 1:  # FOUND high level path
                    self.Root_renew()  # ROOT level down
                    cur_root_level -= 1
                    
                elif cur_root_level == 1:  # FOUND the route at level 1
                    return node.traj

            self.SetNewSubgoal(node_start=node_start, subgoal_traj=subgoal_traj)
            self.delete_child(node_start=node_start, subgoal_traj=subgoal_traj)

            self.success_traj[node.s[0]].add(tuple(node.traj))

        self.backpropagate(node=node, node_start=node_start)  # , reward=reward)

    # Only operate When find the path of high-level
    def delete_child(self, node_start: H_Node_HW, subgoal_traj: list):  
        for action, child in node_start.children.items():
            if child.s == subgoal_traj[0]:
                del node_start.children[action]
                break
            
    # SELECTLEAF in pseudo code
    def selectNode(self, node: H_Node_HW):
        while not node.isTerminal and not node.isCycle:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
                # node.set_subgoals()
            else:
                return self.expand(node)

        return node

    def expand(self, node: H_Node_HW):
    
        
        
        # randomly choose the action from possible action
        action = random.choice(list(node.untried_Actions))

        next_s = node.step(action)

        # Create new node and update the current node
        new_Node = H_Node_HW(s=next_s, env=deepcopy(self.env), parent=node)
        node.untried_Actions.remove(action)
        node.children[action] = new_Node

        # Compare only ***current level***'s children
        self.checkFullExpand(node)

        return new_Node

    def checkFullExpand(self, node: H_Node_HW):
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
        if new_root_level == 1:  # NOT allow ACTION stay (0, 0)
            self.root.env = self.env
            self.root.untried_Actions = self.root.getPossibleActions()
            self.root.traj = [root_state]
            self.root.traj_dict[new_root_level] = [root_state[1:]]

        else:  # allow ACTION stay (0, 0)
            # self.root.env = self.root_env
            self.root.env = self.env

            self.root.untried_Actions = self.root.getPossibleActions()
            self.root.traj_dict[new_root_level] = []

        self.root.isFullyExpanded = False
        self.root.distance = self.root.get_distance()
        self.root.totalReward = 0.0
        self.root.numVisits = 0

    def backpropagate(self, node: H_Node_HW, node_start: H_Node_HW):
        reward = self.getReward(node)
        while True:  # root = parent is None
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
            if node is node_start:
                node.numVisits += 1
                break
            
            reward = self.gamma * reward
            reward += self.getReward(node)

    def getBestChild(self, node: H_Node_HW, explorationValue: float):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            # calculate UCT value
            nodeValue = (
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
    def getAction(self, parent: H_Node_HW, child: H_Node_HW):
        for action, node in parent.children.items():
            if node is child:
                return action

        raise ValueError("there is no relation between parent and child")

    def getReward(self, node: H_Node_HW):
        return self.env.calculate_reward(node=node)

    def delete_cycle(self, node: H_Node_HW):        
        if node.isCycle:
            while len(node.children) == 0 and not node.untried_Actions:
                # Prune the node from its parent
                for action, child in node.parent.children.items():
                    if child.s == node.s:
                        del node.parent.children[action]  # prune parent to node
                        node = node.parent
                        break
        
    # def AssignNextSubgoal(self, ):
    #     b=1
    
    # def Expand_MultiLevel(self, ):
    #     c=1
        
    def SetNewSubgoal(self, node_start: H_Node_HW, subgoal_traj: list):
        node_start.subgoal_set.add(tuple(subgoal_traj))
        
    # Find the start node (Root or Extendable)
    def Backtrace(self, node: H_Node_HW):
        traj = []
        while (not node.isRoot) and (not node.isExtendable):
            traj.insert(0, node.s)  # leaf to extendable, leaf to root
            node = node.parent
        return node, traj        