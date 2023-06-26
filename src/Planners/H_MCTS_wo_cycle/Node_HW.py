from copy import deepcopy

from src.Env.Grid.Higher_Grids_HW import HighLevelGrids3
from src.Env.utils import hierarchy_map


class H_Node_HW:
    def __init__(self, s: tuple, env: HighLevelGrids3, parent=None):
        self.s = s  # state: (level, x, y)
        self.env = env
        self.H_level = self.env.H_level
        self.set_High_state()
        
        self.parent = parent
        self.parent_level()
        
        self.children = dict()  # key: action, value: children
        
        self.set_traj()

        self.numVisits = 0
        self.totalReward = 0.0
        self.achieved_subgoal = []

        self.untried_Actions = deepcopy(self.getPossibleActions())

        self.set_R_status()  # set self.isRoot
        self.set_T_status()  # set self.isTerminal
        self.get_distance()

        # NOT terminal -> CAN have children
        # terminal -> CANNOT have children
        self.isFullyExpanded = self.isTerminal
        
        self.checkCycle()       # set self.isCycle, self.untried_Actions
        self.CheckExtendable()  # set self.isExtendable
        
    def set_traj(self):
        if self.parent is None:  # Root
            if self.s[0] == 1:   # Root at level 1
                self.traj = [self.s]
                self.traj_dict = {self.H_level: [self.s[1:]]}
            
            else:
                self.traj = []
                self.traj_dict = {self.H_level: []}
            
        else:  # non-Root
            self.traj = deepcopy(self.parent.traj)
            self.traj.append(self.s)
            self.traj_dict = deepcopy(self.parent.traj_dict)
            self.set_traj_dict()

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
            self.isTerminal = self.env.check_goal_pos(self.s)

    def get_distance(self):  # at its level, v_{approx}
        self.distance = self.env.calculate_d2Goal(s=self.s)

    def num_child(self):
        return len(self.children)

    def step(self, action):  # -> state
        return self.env.step(self.s, action)

    # Do not consider cycle
    def getPossibleActions(self):
        return self.env.possible_Action_dict[self.s]

    # Check the state belongs Cycle or not
    def checkCycle(self):
        self.isCycle=False
        # if self.s[0] != 1:  # Allow cycle for high level
        #     self.isCycle = False
        # else:  # level at 1
        #     self.untried_Actions = [
        #         action
        #         for action in self.untried_Actions
        #         if self.step(action=action) not in self.traj
        #     ]
        #     if not self.isTerminal:
        #         # No possible Action
        #         self.isCycle = (len(self.untried_Actions) == 0)
        #     else:
        #         self.isCycle = False
    
    # Set high level state
    def set_High_state(self):
        self.level_pos = dict()
        for level in range(self.s[0], self.H_level + 1):
            high_x, high_y = hierarchy_map(level_curr=self.s[0], level2move=level, pos=(self.s[1], self.s[2]))
            self.level_pos[level] = (level, high_x, high_y)
            
    def set_traj_dict(self):  # for high level trajectories
        for level, s in self.level_pos.items():
            if self.parent.isRoot:  # allow STAY
                self.traj_dict[level].append(s)
            else:  # 
                if self.traj_dict[level][-1] != s:  # check the last element of the list
                    self.traj_dict[level].append(s)
    
    def CheckExtendable(self):
        self.subgoal_set = set()
        self.isExtendable = False

        if self.parent is not None:
            for subgoal_traj in self.parent.subgoal_set:
                obj_state = subgoal_traj[0]
                if self.s[0] >= obj_state[0]:
                    continue 
                
                state = self.level_pos[obj_state[0]]
                if state != obj_state:
                    continue
                else:
                    if obj_state[0] == self.s[0] + 1:
                        self.achieved_subgoal.append(subgoal_traj[0])
                        if len(subgoal_traj) > 1:
                            self.isExtendable = True
                            self.expand_untried_Actions(obj_state[0])
                            self.subgoal_set.add(subgoal_traj[1:])
                        
                    elif obj_state[0] > self.s[0] + 1:
                        self.achieved_subgoal.append(subgoal_traj[0])
                        if len(subgoal_traj) > 1:
                            self.subgoal_set.add(subgoal_traj[1:])
                        
                    else:  # obj_state[0] < self.s[0] + 1
                        # raise Exception('wrong subgoal at parent')
                        pass
            
            # No achieved subgoal -> inherit subgoal_set
            if len(self.achieved_subgoal) == 0:
                self.subgoal_set = self.parent.subgoal_set
                    
    # Expand the extendable node's untried actions into high-level actions for Exploration
    def expand_untried_Actions(self, expandLevel: int):
        if expandLevel == 1:
            raise Exception('wrong level input')
        else:  # level > 1
            if not self.isCycle:
                s = self.level_pos[expandLevel]
                possible_A = self.env.get_possible_Action(s)
                self.untried_Actions.extend(possible_A)