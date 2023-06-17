from collections import defaultdict
from abc import ABC, abstractmethod
from copy import deepcopy

import time
import math
import random

from src.Env.Grid.Higher_Grids import HighLevelGrids
from src.Env.utils import *


class H_Node:
    def __init__(self, s: tuple, env: HighLevelGrids, parent=None):
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

        self.untried_Actions = self.getPossibleActions(env)

        self.set_R_status()
        self.set_T_status(env)
        self.get_distance(env)
        self.set_subgoals()

        # is not terminal -> then can have children
        # is terminal -> then cannot have children
        self.isFullyExpanded = self.isTerminal
        self.check_State(Env=env)

    # set Root status
    def set_R_status(self):
        if self.parent is None:
            self.isRoot = True
        else:
            self.isRoot = False

    # Set the node belongs to terminal or not
    # The level is not important
    def set_T_status(self, env: HighLevelGrids):
        if self.isRoot == True:  # Root CANNOT be terminal node
            self.isTerminal = False
        else:  # do not separate level
            self.isTerminal = env.check_goal_pos(self.s)

    def get_distance(self, env: HighLevelGrids):  # at its level, v_{approx}
        self.distance = env.calculate_d2Goal(s=self.s)

    def num_child(self):
        return len(self.children)

    def step(self, env: HighLevelGrids, action):  # -> state
        return env.step(self.s, action)

    def getPossibleActions(self, env: HighLevelGrids):
        return env.get_possible_Action(self.s)

    def set_subgoals(self):
        if self.parent is None:  # Root node
            self.subgoal_set = set()
            return

        # non-Root node
        self.subgoal_set = set()
        for level_subgoal, subgoal_x, subgoal_y in self.parent.subgoal_set:
            if level_subgoal > self.s[0]:
                map_x, map_y = hierarchy_map(
                    level_current=self.s[0],
                    level2move=level_subgoal,
                    pos=(self.s[1], self.s[2]),
                )
                if (subgoal_x, subgoal_y) != (map_x, map_y):  # belong subgoal check
                    self.subgoal_set.add((level_subgoal, subgoal_x, subgoal_y))
                    
                #else:  # extendable, for exploration add the unexisted subgoal
                    #self.subgoal_set.add(())
                    

    # only implement at init
    def check_State(self, Env: HighLevelGrids):
        if self.s[0] != 1:
            self.isCycle = False
            return
        self.untried_Actions = [
            action
            for action in self.untried_Actions
            if self.step(env=Env, action=action) not in self.traj
        ]
        if not self.isTerminal:
            self.isCycle = (
                len(self.untried_Actions) == 0
            )  # cannot try action (belong to trajectory)
        else:
            self.isCycle = False


class H_MCTS:
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=1,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=(-1) * 4,
        cycle_penalty=(-1) * 100,
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
            cycle_penalty,
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
        action_cost=(-1) * 4,
        cycle_penalty=(-1) * 100,
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
            cycle_penalty,
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
            cycle_penalty,
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

        if best_traj[-1][0] == 1:
            if (best_traj[-1][1], best_traj[-1][2]) == self.Env.goal_dict[
                1
            ]:  # find feasible path
                return best_traj, root_Performance, True

        return best_traj, root_Performance, False

    # One Simulation
    def executeRound(self):
        curr_level = self.root.s[0]

        # Select Leaf
        node = self.selectNode(self.root)

        # Root go low level and set subgoal
        if node.isTerminal:
            if (
                len(self.success_traj[node.s[0]]) == 0
                and node.s[0] == curr_level
                and curr_level != 1
            ):
                self.Root_renew()  # level down
                self.delete_child(target_state=node.traj[0])
                curr_level -= 1

            if node.s[0] > curr_level:
                
                self.root.subgoal_set.update(node.traj)
                self.delete_child(target_state=node.traj[0])

            self.success_traj[node.s[0]].add(tuple(node.traj))

        # Expand Multi Level

        self.backpropagate(node=node)  # , reward=reward)

    def delete_child(self, target_state):
        for A, child in list(self.root.children.items()):
            if child.s == target_state:
                del self.root.children[A]
                break

    # SELECTLEAF in pseudo code
    def selectNode(self, node: H_Node):
        while not node.isTerminal and not node.isCycle:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
                node.set_subgoals()
            else:
                return self.expand(node)

        return node

    def expand(self, node: H_Node):
        # randomly choose the action from possible action
        action = random.choice(list(node.untried_Actions))
        env = self.root_Env if node.isRoot and node.s[0] > 1 else self.Env

        next_s = node.step(env, action)

        # Create new node and update the current node
        new_Node = H_Node(s=next_s, env=deepcopy(self.Env), parent=node)
        node.untried_Actions.remove(action)
        node.children[action] = new_Node

        # Compare only ***current level***'s children
        self.checkFullExpand(node, env)

        return new_Node

    def checkFullExpand(self, node: H_Node, env: HighLevelGrids):
        if len(node.untried_Actions) == 0:
            node.isFullyExpanded = True

    def Root_renew(self):  # root to low level
        new_root_level = self.root.s[0] - 1
        root_state = (
            new_root_level,
            self.Env.start_dict[new_root_level][0],
            self.Env.start_dict[new_root_level][1],
        )
        self.root.s = root_state
        if new_root_level == 1:  # NOT allow ACTION stay (0, 0)
            self.root.untried_Actions = self.root.getPossibleActions(self.Env)
            self.root.traj = [root_state]
        else:  # allow ACTION stay (0, 0)
            self.root.untried_Actions = self.root.getPossibleActions(self.root_Env)

        self.root.isFullyExpanded = False
        self.root.distance = self.root.get_distance(self.Env)
        self.root.totalReward = 0.0
        self.root.numVisits = 0

    def backpropagate(self, node: H_Node):
        reward = self.getReward(node)
        while node is not None:  # root = parent is None
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
            if node is not None and not node.isRoot:
                reward += self.getReward(node)

    def getBestChild(self, node: H_Node, explorationValue: float):
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
    def getAction(self, parent: H_Node, child: H_Node):
        for action, node in parent.children.items():
            if node is child:
                return action

        raise ValueError("there is no relation between parent and child")

    def getReward(self, node: H_Node):
        return self.Env.calculate_reward(node=node, subgoal_set=node.parent.subgoal_set)
    
    # Not prefer....
    def update_subgoal_pair(self, traj):
        for i, j in zip(traj[:-1], traj[1:]):
            self.root.subgoal_set.update((i, j))
            

    # def checkFailure(self, node):

    # def extendable_subgoal(self, node: H_Node):
    #     # means arrive at subgoal point
    #     if node.high_level_terminal is True and node.level1_terminal is False:
    #         # give reward and give high-level node for exploration
    #         node.
