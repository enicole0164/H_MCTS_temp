from copy import deepcopy

import math
import random

from src.Env.Grid.Higher_Grids import HighLevelGrids
from src.Planners.H_MCTS_add.Node import H_Node


class H_MCTS:
    def __init__(
        self,
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=1,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=(-1) * 2,
        cycle_penalty=(-1) * 50,
        iter_Limit=10000,
        explorationConstant=1 / math.sqrt(2),  # 1 / math.sqrt(2)
        random_seed=25,
        num_barrier=15,
        gamma=1,
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
        grid_setting,  # l1_rows, l1_cols, l1_width, l1_height
        H_level=None,
        A_space={(1, 0), (-1, 0), (0, 1), (0, -1)},
        RS=2,
        l1_goal_reward=10,
        l1_subgoal_reward=2,
        action_cost=(-1) * 2,
        cycle_penalty=(-1) * 20,
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

        # only high level(>= 2) root can do action with stay (0, 0)
        # self.root_env = deepcopy(self.env)
        # self.root_env.A_space.add((0, 0))
        # self.root_env.set_possible_Action_dict()

        # Generate initial state at the highest level
        self.init_s = (
            self.env.H_level,
            self.env.start_dict[self.env.H_level][0],
            self.env.start_dict[self.env.H_level][1],
        )

    def set_Root(self):
        if self.init_s[0] != 1:
            # self.root = H_Node(self.init_s, deepcopy(self.root_env), parent=None)
            self.root = H_Node(self.init_s, deepcopy(self.env), parent=None)

        else:  # only solve at level 1
            self.root = H_Node(self.init_s, deepcopy(self.env), parent=None)

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
            if (best_traj[-1][1], best_traj[-1][2]) == self.env.goal_dict[
                1
            ]:  # find feasible path
                return best_traj, root_Performance, True

        return best_traj, root_Performance, False

    # One Simulation
    def executeRound(self):
        cur_root_level = self.root.s[0]

        # Select Leaf
        node = self.selectNode(self.root)

        if node.isTerminal:
            # Root go low level and set subgoal
            if len(self.success_traj[node.s[0]]) == 0 and cur_root_level != 1:
                self.Root_renew()  # level down
                cur_root_level -= 1

            if node.s[0] > cur_root_level:
                print("subgoal Increased, {}".format(node.traj))

                self.root.subgoal_set.update(node.traj)
                pruned_node = self.delete_child(target_state=node.traj[0])

                # print(pruned_node.s)

            self.success_traj[node.s[0]].add(tuple(node.traj))

        # Expand Multi Level

        self.backpropagate(node=node)  # , reward=reward)

    def delete_child(self, target_state):
        for A, child in list(self.root.children.items()):
            if child.s == target_state:
                pruned_node = self.root.children[A]
                del self.root.children[A]
                return pruned_node

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

        next_s = node.step(action)

        # Create new node and update the current node
        new_Node = H_Node(s=next_s, env=deepcopy(self.env), parent=node)
        node.untried_Actions.remove(action)
        node.children[action] = new_Node

        # Compare only ***current level***'s children
        self.checkFullExpand(node)

        return new_Node

    def checkFullExpand(self, node: H_Node):
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
        return self.env.calculate_reward(node=node, subgoal_set=node.parent.subgoal_set)

    # def checkFailure(self, node):

    # def extendable_subgoal(self, node: H_Node):
    #     # means arrive at subgoal point
    #     if node.high_level_terminal is True and node.level1_terminal is False:
    #         # give reward and give high-level node for exploration
    #         node.
