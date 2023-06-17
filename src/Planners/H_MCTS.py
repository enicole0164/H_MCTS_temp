from collections import defaultdict
from abc import ABC, abstractmethod
from copy import deepcopy

import time
import math
import random

from src.Env.Grid.Higher_Grids import HighLevelGrids
from src.Env.utils import *


class H_Node:
    def __init__(self, s, Grid: HighLevelGrids, parent=None):
        self.s = s  # (level, x, y)
        # self.Env = Grid
        self.parent = parent
        self.children = dict()  # key: action, value: children

        if parent is not None:  # non-Root
            self.traj = deepcopy(parent.traj)
            self.traj.append(s)
        else:  # Root
            self.traj = []

        self.numVisits = 0
        self.totalReward = 0.0

        self.subgoal_set = set()  # set of subgoal with state (level, x, y)

        self.untried_Actions = self.getPossibleActions(Grid)

        self.set_R_status()
        self.set_T_status(Grid)
        self.get_distance(Grid)
        self.set_subgoals()

        # is not terminal -> then can have children
        # is terminal -> then cannot have children
        self.isFullyExpanded = self.isTerminal

    # set Root status
    def set_R_status(self):
        if self.parent is None:
            self.isRoot = True
        else:
            self.isRoot = False

    # Set the node belongs to terminal or not
    # The level is not important
    def set_T_status(self, Grid: HighLevelGrids):
        if self.isRoot == True:  # Root CANNOT be terminal node
            self.isTerminal = False
        else:
            # do not separate level
            self.isTerminal = Grid.check_goal_pos(self.s)

    def get_distance(self, Grid: HighLevelGrids):  # at its level, v_{approx}
        self.distance = Grid.calculate_d2Goal(s=self.s)

    # only for Root node. When find the level i's feasible path.
    # generate the Root node state change. level i to i-1

    def num_child(self):
        return len(self.children)

    def step(self, Grid: HighLevelGrids, action):  # -> state
        return Grid.step(self.s, action)

    def getPossibleActions(self, Grid: HighLevelGrids):
        return Grid.get_possible_Action(self.s)

    def set_subgoals(self):
        if self.parent is None:  # Root node
            pass
        else:  # non-Root node
            self.subgoal_set = {i for i in self.parent.subgoal_set if i[0] > self.s[0]}
            copy_subgoal = deepcopy(self.subgoal_set)
            for level_subgoal, subgoal_x, subgoal_y in copy_subgoal:
                map_x, map_y = hierarchy_map(
                    level_current=self.s[0],
                    level_to_move=level_subgoal,
                    x=self.s[1],
                    y=self.s[2],
                )

                if (subgoal_x, subgoal_y) == (map_x, map_y) and level_subgoal > self.s[0]:
                    self.subgoal_set.remove((level_subgoal, map_x, map_y))


class H_MCTS:
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=1,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=-1,
        iter_Limit=10000,
        explorationConstant=1 / math.sqrt(2),  # 1 / math.sqrt(2)
        random_seed=25,
        informed=False,
        l1_barrier=True,
        num_barrier=15,
    ):
        self.searchLimit = iter_Limit
        self.limitType = "iter"
        self.iteration = 0

        self.l1_rows = grid_setting[0]
        self.l1_cols = grid_setting[1]
        self.l1_width = grid_setting[2]
        self.l1_height = grid_setting[3]
        self.A_space = A_space
        self.RS = RS

        self.explorationConstant = explorationConstant

        self.set_Env(
            grid_setting,
            H_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
            random_seed,
            l1_barrier,
            num_barrier,
        )

        # Assume that we know the Env
        self.informed = informed
        self.success_traj = {level + 1: set() for level in range(self.Env.H_level)}

    # Set Environment for Root node and children nodes
    # Have difference at action space
    def set_Env(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=-1,
        random_seed=26,
        l1_barrier=True,
        num_barrier=10,
    ):
        self.Env = HighLevelGrids(
            grid_setting,
            H_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
            random_seed,
            l1_barrier,
            num_barrier,
        )

        # only high level(>= 2) root can do action with stay (0, 0)
        self.root_Env = HighLevelGrids(
            grid_setting,
            H_level,
            A_space
            | {(0, 0)},  # only Root can have action stay (because of abstraction)
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
            random_seed,
            l1_barrier,
            num_barrier,
        )

        # Generate initial state at the highest level
        self.init_s = (
            self.Env.H_level,
            self.Env.start_dict[self.Env.H_level][0],
            self.Env.start_dict[self.Env.H_level][1],
        )

    def set_Root(self):
        if self.init_s[0] != 1:
            self.root = H_Node(self.init_s, deepcopy(self.root_Env), parent=None)
        else:
            self.root = H_Node(self.init_s, deepcopy(self.Env), parent=None)

    # while loop in pseudo code (replace by for loop)
    def search(self):
        root_Performance = []
        self.set_Root()  # set root and its subgoal ( destination at high level)

        for i in range(self.searchLimit):
            self.executeRound()
            root_Performance.append(self.root.totalReward / self.root.numVisits)

        node = self.root
        best_traj = [node.s]
        while len(node.children.keys()) != 0:
            node = self.getBestChild(node, 0)
            best_traj.append(node.s)
        return best_traj, root_Performance

    # One Simulation
    def executeRound(self):
        curr_level = self.root.s[0]

        
        # Select Leaf
        node = self.selectNode(self.root)

        ######################## Check the failure############################# 
        
        
        
        ######################################################################
        # Root go low level and set subgoal
        if node.isTerminal:
            if len(self.success_traj[node.s[0]]) == 0:
                # print(f"trajectory to goal is {node.traj}")                
                if node.s[0] == curr_level and curr_level != 1:                    
                    self.Root_renew()
                    for A, child in list(self.root.children.items()):
                        if child.s == node.traj[0]:
                            del self.root.children[A]
                            break
                
            if node.s[0] > curr_level:
                self.root.subgoal_set.update(node.traj)
                for A, child in list(self.root.children.items()):
                    if child.s == node.traj[0]:
                        del self.root.children[A]
                        break
                    
            self.success_traj[node.s[0]].add(tuple(node.traj))

        # Expand Multi Level

        self.backpropagate(node=node)  # , reward=reward)
        
    # SELECTLEAF in pseudo code
    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
                node.set_subgoals()
            else:
                return self.expand(node)

        return node

    def expand(self, node: H_Node):
        # randomly choose the action from possible action
        # untried_actions = node.getPossibleActions()
        action = random.choice(list(node.untried_Actions))
        if node.isRoot:
            next_s = node.step(self.root_Env, action)
        else:
            next_s = node.step(self.Env, action)

        new_Node = H_Node(s=next_s, Grid=deepcopy(self.Env), parent=node)
        node.untried_Actions.remove(action)

        node.children[action] = new_Node

        # Compare only current level's children
        num_children = len(
            [ch for ch in node.children.values() if ch.s[0] == node.s[0]]
        )
        num_actions = len(node.getPossibleActions(self.Env))
        if node.isRoot and node.s[0] > 1:
            num_actions = len(node.getPossibleActions(self.root_Env))

        if num_children == num_actions:
            node.isFullyExpanded = True

        return new_Node

    def Root_renew(self):  # root to low level
        new_root_level = self.root.s[0] - 1
        self.root.s = (
            new_root_level,
            self.Env.start_dict[new_root_level][0],
            self.Env.start_dict[new_root_level][1],
        )
        if new_root_level == 1:
            self.root.untried_Actions = self.root.getPossibleActions(self.Env)
        else:
            self.root.untried_Actions = self.root.getPossibleActions(self.root_Env)

        self.root.isFullyExpanded = False
        self.root.distance = self.root.get_distance(self.Env)
        self.root.totalReward = 0.0
        self.root.numVisits = 0
        
        
            

    def backpropagate(self, node):
        reward = self.getReward(node)
        while node is not None:  # root = parent is None
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
            if node is not None and not node.isRoot:
                reward += self.getReward(node)

    def getBestChild(self, node, explorationValue):
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
    def getAction(self, parent, child):
        for action, node in parent.children.items():
            if node is child:
                return action

        raise ValueError("there is no relation between parent and child")

    def getReward(self, node):
        return self.Env.calculate_reward(
            node=node, subgoal_set=node.parent.subgoal_set
        )
        
    # def checkFailure(self, node):
        

    # def extendable_subgoal(self, node: H_Node):
    #     # means arrive at subgoal point
    #     if node.high_level_terminal is True and node.level1_terminal is False:
    #         # give reward and give high-level node for exploration
    #         node.
