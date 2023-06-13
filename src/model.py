from collections import defaultdict
from abc import ABC, abstractmethod

import time
import math
import random

from src.Env.Grid.Higher_Grids import high_level_Grids
from Env.utils import *


class H_Node:
    def __init__(self, state, Grid: high_level_Grids, parent=None):
        self.state = state  # (level, x, y)
        # self.level = level  # heirarchical level
        # self.x = x  # horizontal position at grid
        # self.y = y  # vertical position at grid
        self.Env = Grid
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.curr_subgoal_set = set()  # set of subgoal with state (level, x, y)
        self.total_subgoal_set = set()  # for not adding at extenable subgoals
        self.destination = None
        self.children = dict()

        self.set_Root_status_at_Level()
        self.set_Terminal_status()
        # self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal

    def __str__(self):
        return "(level: {}, x: {}, y: {}): (action={}, visits={}, reward={:d}, ratio={:0.4f})".format(
            self.state[0],
            self.state[1],
            self.state[2],
            self.action,
            self.num_visits,
            self.total_simulation_reward,
            self.performance,
        )

    def set_Root_status_at_Level(self):
        # Require both no parent and position
        if self.parent is None and self.Env.check_Root_pos_at_Level(self.state):
            self.is_Root = True

        # cycle
        elif self.parent is not None and self.Env.check_Root_pos_at_Level(self.state):
            self.is_Root = False
        else:
            self.is_Root = self.Env.check_Root_pos_at_Level(self.state)

    def set_Terminal_status(self):
        # Root CANNOT be terminal node (even at same cell)
        # at least have a move (include stay)
        if self.is_Root == True:
            self.isTerminal_at_Level = False
            self.isTerminal = False
        else:
            self.isTerminal_at_Level = self.Env.check_termination_pos_at_Level(
                self.state
            )
            if self.level == 1:
                self.isTerminal = self.Env.check_termination_pos_at_Level(self.state)
            else:
                self.isTerminal = False

    def go_low_level(self):
        if self.is_Root:
            level = self.state[0] - 1
            self.state = (
                level,
                self.Env.start_dict[level][0],
                self.Env.start_dict[level][1],
            )

    def num_child(self):
        return len(self.children)

    def step(self, action):
        return self.Env.step(self.state, action, self.curr_subgoal_set)

    def getPossibleActions(self):
        return self.Env.get_possible_Action(self.state)


class H_MCTS:
    def __init__(
        self,
        l1_rows,
        l1_cols,
        l1_width,
        l1_height,
        destination_radius=1,
        barrier_find_segment=101,
        highest_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=20,
        l1_subgoal_reward=4,
        action_cost=-1,
        timeLimit=None,
        iter_Limit=None,
        explorationConstant=1 / math.sqrt(2),
    ):
        if iter_Limit is not None:
            self.searchLimit = iter_Limit
            self.limitType = "iter"

        else:  # iter_Limit is None
            if timeLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            else:
                self.searchLimit = timeLimit
                self.limitType = "time"

        self.iteration = 0

        self.set_Env(
            l1_rows,
            l1_cols,
            l1_width,
            l1_height,
            destination_radius,
            barrier_find_segment,
            highest_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
        )

    def set_Env(
        self,
        l1_rows,
        l1_cols,
        l1_width,
        l1_height,
        destination_radius=1,
        barrier_find_segment=101,
        highest_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=20,
        l1_subgoal_reward=4,
        action_cost=-1,
    ):
        self.Env = high_level_Grids(
            l1_rows,
            l1_cols,
            l1_width,
            l1_height,
            destination_radius,
            barrier_find_segment,
            highest_level,
            A_space,
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
        )
        self.root_Env = high_level_Grids(
            l1_rows,
            l1_cols,
            l1_width,
            l1_height,
            destination_radius,
            barrier_find_segment,
            highest_level,
            A_space.add((0, 0)),
            RS,
            l1_goal_reward,
            l1_subgoal_reward,
            action_cost,
        )

        level = self.Env.highest_level
        self.initial_state = (
            level,
            self.Env.start_dict[level][0],
            self.Env.start_dict[level][1],
        )

    def set_Root(self):
        self.root = H_Node(self.initial_state, self.root_Env, parent=None)
        self.root.curr_subgoal_set.add(
            (
                self.root.Env.highest_level,
                self.root.Env.dest_dict[0],
                self.root.Env.dest_dict[1],
            )
        )
        self.root.total_subgoal_set.add(
            (
                self.root.Env.highest_level,
                self.root.Env.dest_dict[0],
                self.root.Env.dest_dict[1],
            )
        )

    def search(self):
        self.root = H_Node(self.initial_state, self.root_Env, parent=None)

        if self.limitType == "time":
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:  # limitType: iterations
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        # reward =

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node: H_Node):
        actions = node.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                next_state, r, done, total_done, distance = node.step(action)
                newNode = H_Node(state=next_state, Grid=self.Env, parent=node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

        raise Exception("Should never reach here")

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
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

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
