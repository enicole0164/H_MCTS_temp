from copy import deepcopy

import math
import random
import sys

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
        l1_goal_reward=16,
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
        extend_zero=0.005,
    ):
        self.searchLimit = iter_Limit
        self.limitType = "iter"
        self.iteration = 0

        self.gamma = gamma

        # Set alpha
        self.alpha = alpha
        self.constant_c = constant_c
        self.extend_zero = extend_zero

        self.level1expand_cnt = 0

        self.l1_rows, self.l1_cols = grid_setting[0], grid_setting[1]
        self.l1_width, self.l1_height = grid_setting[2], grid_setting[3]
        self.A_space = A_space
        self.RS = RS
        self.explorationConstant_h = explorationConstant_h
        self.explorationConstant_l = explorationConstant_l

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
        )

        # Assume that we know the env
        self.success_traj = {level: set() for level in self.env.levels}

        # key: (state in level 1 or 2)
        # value: (action -> [next_s, cnt])
        # Example: (1, 1, 1) -> {(1,0,1) -> [(1, 1, 2), 1], (1,0,-1) ->[(1, 1, 0), 2], (1,1,0) -> [(1, 2, 0), 3], (1,-1,0) -> [(1, 0, 1), 4]}
        self.extend_cnt = {}

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
    ):  
        self.cont_env = Continuous_Grid(
            grid_settings=grid_setting[:4],
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
            goal_n_start_distance=grid_setting[4],
        )
        
        self.env = HighLevelGrids2(
            grid_settings=grid_setting,
            H_level=H_level,
            random_seed=random_seed,
            num_barrier=num_barrier,
        )
        
        self.env.inherit_start_goal(self.cont_env.start_dict, self.cont_env.goal_dict)

        # Generate initial state at the highest level
        self.init_s = (
            self.env.H_level,
            self.env.start_dict[self.env.H_level][0],
            self.env.start_dict[self.env.H_level][1],
        )

    def set_Root(self):
        self.root = H_Node_Cont(self.init_s, deepcopy(self.env), parent=None, extendable=True, extendable_at_level0=True)

    # while loop in pseudo code (replace by for loop)
    def search(self, tree_save_path=None, traj_save_path=None):
        # print("In search, ...")
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
                # print("DRAW TREE")
                # self.draw_tree(self.root)
                if traj_save_path:
                    sys.stdout.close()
                    sys.stdout = stdout_fileno
                return path, success, i + 1
        # print("FAIL DRAW TREE")
        # self.draw_tree(self.root)
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
        # # Delete Node if Cycle
        # while node.s[0] > 0 and node.isCycle:
        #     self.delete_cycle(node)
        #     node = self.selectNode(self.root)
        #     node_start, subgoal_traj = self.Backtrace(node)

        # Found the path
        if node.isTerminal:
            print("FOUND PATH IN LEVEL ",subgoal_traj[0][0] , subgoal_traj)
            # if not node_start.isRoot:
            #     print("DRAW TREE")
            #     self.draw_tree(self.root)
            print("Node_start", node_start.s)
            if len(self.success_traj[node.s[0]]) == 0:  # first path at level node.s[0]
                if cur_root_level > 0:  # FOUND high level path
                    self.Root_renew()  # ROOT level down
                    cur_root_level -= 1

                elif cur_root_level == 0:  # FOUND the route at level 1
                    return node.traj

            # Reset subgoal and delete subgoal route
            self.SetNewSubgoal(node_start=node_start, subgoal_traj=subgoal_traj)
            # Extended Result
            if not node_start.isRoot:
                # print("BEFORE")
                # self.draw_tree(node_start)
                # Recursively add new subgoal to children of node_start
                self.TraverseNAddNewSubgoal(node_start, node_start, subgoal_traj=subgoal_traj)
                # print("AFTER")
                # self.draw_tree(node_start)
            self.delete_child(node_start=node_start, subgoal_traj=subgoal_traj)

            self.success_traj[node.s[0]].add(tuple(node.traj))

            # if not node_start.isRoot:
            #     print("DRAW TREE")
            #     self.draw_tree(self.root)

        # self.backpropagate(node=node, node_start=self.root)
        self.backpropagate(node=node, node_start=node_start)
        
    # Only operate When find the path of high-level
    def delete_child(self, node_start: H_Node_Cont, subgoal_traj: list):
        for action, child in node_start.children.items():
            if child.s == subgoal_traj[0]:
                del node_start.children[action]
                break

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

                # Exploitation
                if len(node.children) >= expand_limit:
                    node = self.getBestChild(node, self.explorationConstant_l)
                # Explortation
                else:
                    return self.expand(node)
            else:
                self.level1expand_cnt += 1
                return self.expand(node)

        return node

    def expand(self, node: H_Node_Cont):
        # print("expand")
        if node.s[0] > 0:
            # # Previous
            # action = random.choice(list(node.untried_Actions))
            # next_s = node.step(action)
            # Now
            if not node.isRoot and node.parent.s[0] == node.s[0]:
                exclude_Action = (node.s[0], node.parent.s[1] - node.s[1], node.parent.s[2] - node.s[2])
                if exclude_Action in node.untried_Actions:
                    node.untried_Actions.remove(exclude_Action)
                action = random.choice(list(node.untried_Actions))
                next_s = node.step(action)
            if not node.isRoot and node.parent.s[0] == node.s[0] - 1:
                # print(node.parent.level_pos)
                # print(node.parent.s[0])
                # print(node.parent.s)
                # print(self.root.s)
                # print(node.s)
                high_parent = node.parent.level_pos[node.s[0]]
                exclude_Action = (node.s[0], high_parent[1] - node.s[1], high_parent[2] - node.s[2])
                if exclude_Action in node.untried_Actions:
                    node.untried_Actions.remove(exclude_Action)
                action = random.choice(list(node.untried_Actions))
                next_s = node.step(action)
            else:
                action = random.choice(list(node.untried_Actions))
                next_s = node.step(action)
        else: # node.s[0] == 0
            # # Previous
            # action = node.getPossibleAction()
            # next_s = node.step(action)
            # Expand from 0 too
            w = random.uniform(0, 1)

            node_level1_s = node.level_pos[1]
            if node.parent != None:
                if node.parent.level_pos == None:
                    assert(False)
                parent_level1_s = node.parent.level_pos[1]
                # Remove going back to parent
                going_back_to_parent = (node_level1_s[0], parent_level1_s[1] - node_level1_s[1], parent_level1_s[2] - node_level1_s[2])
                if going_back_to_parent in node.untried_Actions:
                    node.untried_Actions.remove(going_back_to_parent)

            if w > self.extend_zero or len(node.untried_Actions) == 0:
                action = node.getPossibleAction()
                next_s = node.step(action)
            else:
                # # Trial 1
                # if not 1 in node.level_pos.keys():
                #     print(node)
                #     action = node.getPossibleAction()
                #     next_s = node.step(action)
                # else:
                node_level1_s = node.level_pos[1]

                if node_level1_s not in self.extend_cnt.keys():
                    self.extend_cnt[node_level1_s] = {}

                untried_Action_in_whole = set()
                for action in node.untried_Actions:
                    if not action in self.extend_cnt[node_level1_s].keys():
                        untried_Action_in_whole.add(action)
                
                if len(untried_Action_in_whole) != 0:
                    # print("Or this?")
                    action = random.choice(list(untried_Action_in_whole))
                    action_level = action[0]
                    higher_level_s = node.level_pos[action_level]
                    next_s = (action_level, higher_level_s[1]+action[1], higher_level_s[2]+action[2])

                    self.extend_cnt[node_level1_s][action] = [next_s, 1]
                
                else:
                    # print("This?")
                    cnt = math.inf
                    next_s_n_action_set = set()
                    for a, [n_s, s_cnt] in self.extend_cnt[node_level1_s].items():
                        if a in node.untried_Actions and cnt >= s_cnt:
                            next_s_n_action_set.add((n_s, a))
                    state_n_action = random.choice(list(next_s_n_action_set))
                    next_s, action = state_n_action
                    self.extend_cnt[node_level1_s][action][1] += 1
                    if node_level1_s == (1, 0, 0):
                        print(action, next_s)


                # # ----------------------- Previous -----------------------
                # action = random.choice(list(node.untried_Actions))
                # node.extended_level_0 += 1
                # # measure action_level
                # # print("action",action)
                # # print("node.untried_Actions",node.untried_Actions)
                # action_level = action[0]
                # if action_level not in node.level_pos.keys():
                #     assert(False)
                #     action = node.getPossibleAction()
                #     next_s = node.step(action)
                # else:
                #     higher_level_s = node.level_pos[action_level]
                #     higher_level_node = H_Node_Cont(s=higher_level_s, env=deepcopy(self.env), parent=None)
                #     next_s = higher_level_node.step(action)
                # # ----------------------- Previous -----------------------

        if action[0] > 0:   
            new_Node = H_Node_Cont(s=next_s, env=deepcopy(self.env), parent=node, extendable=True, extendable_at_level0=True)
            node.untried_Actions.remove(action)
        else:
            new_Node = H_Node_Cont(s=next_s, env=deepcopy(self.cont_env), parent=node, extendable=True, extendable_at_level0=True)

        
        node.children[action] = new_Node

        # Compare only ***current level***'s children
        if node.s[0] > 0:
            self.checkFullExpand(node)
            # print(new_Node.s)
            # print(new_Node.untried_Actions)
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
            # self.root.env = self.root_env
            self.root.env = self.env

            self.root.untried_Actions = deepcopy(self.root.getPossibleActions())
            self.root.traj_dict[new_root_level] = []

        self.root.isFullyExpanded = False
        self.root.distance = self.root.get_distance()
        self.root.totalReward = 0.0
        self.root.numVisits = 0

        # JR add
        self.root.level_pos[new_root_level] = root_state

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
                # node.totalReward += reward
                break

            reward = self.gamma * reward
            if node.s[0] != 0:
                reward += self.getReward(node)

    def backpropagate_wo_numVisits(self, node: H_Node_Cont, node_start: H_Node_Cont):
        reward = self.getReward(node)

        if node is node_start:
            node.totalReward += reward
            return

        while True:  # root = parent is None
            node.totalReward += reward
            node = node.parent
            if node is node_start:  # if node is self.root:
                # node.totalReward += reward
                break

            reward = self.gamma * reward
            if node.s[0] != 0:
                reward += self.getReward(node)

    def getBestChild(self, node: H_Node_Cont, explorationValue: float):
        # We can set two exploration constant for each level
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

    # def delete_cycle(self, node: H_Node_Cont):
    #     if node.isCycle:
    #         while (
    #             not [child for child in node.children.values() if child.s[0] == 1]
    #         ) and (not [a for a in node.untried_Actions if a[0] == 1]):
    #             # Prune the node from its parent
    #             for action, child in node.parent.children.items():
    #                 if child == node:
    #                     del node.parent.children[action]  # prune parent to node
    #                     node = node.parent
    #                     break
                
    # When find the high-level path
    def SetNewSubgoal(self, node_start: H_Node_Cont, subgoal_traj: list):
        node_start.subgoal_set.add(tuple(subgoal_traj))

    def TraverseNAddNewSubgoal(self, node_start: H_Node_Cont, node: H_Node_Cont, subgoal_traj: list, subgoal_following=False):
        # node_start의 자손들에게 traverse&addnewsubgoal
        # subgoal_traj를 추가한다.
        for child in node.children.values():
            # 1. Check if child has reached any subgoal already
            if len(child.achieved_subgoal) != 0:
                continue
            # 2. Child hasn't achieved any subgoal
            # Check if child has achieved the subgoal_traj
            subgoal_level = subgoal_traj[0][0]
            if child.s[0] >= subgoal_level:
                continue
            high_level_child_s = child.level_pos[subgoal_level]
            if high_level_child_s == subgoal_traj[0]:
                # Child has achieved THE subgoal!
                child.achieved_subgoal.append(subgoal_traj[0])
                self.backpropagate_wo_numVisits(child, node_start)
                subgoal_following = True
                # Remove all the subgoals having same level with subgoal_traj
                subgoal_to_remove = set()
                for subgoal_seq in child.subgoal_set:
                    if len(subgoal_seq) != 0 and subgoal_seq[0][0] == subgoal_level:
                            subgoal_to_remove.add(subgoal_seq)
                for subgoal_seq in subgoal_to_remove:
                    child.subgoal_set.remove(subgoal_seq)
                # Add the subgoal
                child.subgoal_set.add(tuple(subgoal_traj[1:]))
                self.TraverseNAddNewSubgoal(node_start, child, subgoal_traj[1:], subgoal_following)
            else:
                if not subgoal_following:
                    # Child has opportunity
                    child.subgoal_set.add(tuple(subgoal_traj))
                    self.TraverseNAddNewSubgoal(node_start, child, subgoal_traj, subgoal_following)
                else:
                    # Remove all the subgoals having same level with subgoal_traj
                    subgoal_to_remove = set()
                    for subgoal_seq in child.subgoal_set:
                        if len(subgoal_seq) != 0 and subgoal_seq[0][0] == subgoal_level:
                                subgoal_to_remove.add(subgoal_seq)
                    for subgoal_seq in subgoal_to_remove:
                        child.subgoal_set.remove(subgoal_seq)
                    # Add the subgoal
                    child.subgoal_set.add(tuple(subgoal_traj))
                    self.TraverseNAddNewSubgoal(node_start, child, subgoal_traj, subgoal_following)

                
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
        print(indent + "|-- " , node.achieved_subgoal)
        print(indent + "|-- depth", depth)
        
        # Recursively draw children
        for index, child in enumerate(node.children.values()):
            if index == len(node.children) - 1:
                print(indent + "    ")
                self.draw_tree(child, depth+1)
            else:
                print(indent + "|")
                self.draw_tree(child, depth+1)
