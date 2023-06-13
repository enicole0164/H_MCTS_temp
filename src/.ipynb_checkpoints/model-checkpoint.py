from collections import defaultdict
from abc import ABC, abstractmethod

import time
import math
import random


class H_Node():
    def __init__(self, level, x, y, parent):
        self.level = level  # heirarchical level
        self.x = x  # horizontal position at grid
        self.y = y  # vertical position at grid
        self.is_Root = check_Root(level, x, y)
        self.isTerminal = check_terminal(level, x, y)
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.subgoals = []  # list of subgoal with (level, x, y)
        self.destination = None
        self.children = {}

    def __str__(self):
        return "(level: {}, x: {}, y: {}): (action={}, visits={}, reward={:d}, ratio={:0.4f})".format(
            self.level,
            self.x,
            self.y,
            self.action,
            self.num_visits,
            self.total_simulation_reward,
            self.performance)


def check_Root(level, x, y):
    
    
def check_terminal(level, x, y):
    
    
