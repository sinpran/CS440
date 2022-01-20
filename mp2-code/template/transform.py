
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    if (len(arm.getArmLimit()) >= 1):
        alpha = arm.getArmLimit()[0]
        start_alpha = arm.getArmAngle()[0]
        if (len(arm.getArmLimit()) >= 2):
            beta = arm.getArmLimit()[1]
            start_beta = arm.getArmAngle()[1]
            if (len(arm.getArmLimit()) == 3):
                gamma = arm.getArmLimit()[2]
                start_gamma = arm.getArmAngle()[2]
            else:
                gamma = (0,1)
                start_gamma = 0
        else:
            beta = (0,1)
            gamma = (0,1)
            start_beta = 0
            start_gamma = 0

    rows = int((alpha[1] - alpha[0])/granularity) + 1
    columns = int((beta[1] - beta[0])/granularity) + 1
    third_axis = int((gamma[1] - gamma[0])/granularity) + 1
    offsets = [alpha[0], beta[0], gamma[0]]
    start = (start_alpha, start_beta, start_gamma)

    map = [[[SPACE_CHAR for i in range(third_axis)] for j in range(columns)] for k in range(rows)]
    for a in range(alpha[0], alpha[1] + 1, granularity):
        for b in range(beta[0], beta[1] + 1, granularity):
            for g in range(gamma[0], gamma[1] + 1, granularity):
                arm.setArmAngle((a, b, g))
                idx = angleToIdx((a, b, g), offsets, granularity)
                x = idx[0]
                y = idx[1]
                z = idx[2]
                if doesArmTouchObjects(arm.getArmPosDist(), obstacles, isGoal=False):
                    map[x][y][z] = WALL_CHAR
                    continue
                if doesArmTouchObjects(arm.getArmPosDist(), goals, isGoal=True):
                    map[x][y][z] = WALL_CHAR
                if not isArmWithinWindow(arm.getArmPos(), window):
                    map[x][y][z] = WALL_CHAR
                if doesArmTipTouchGoals(arm.getEnd(), goals):
                    map[x][y][z] = OBJECTIVE_CHAR

    start_idx = angleToIdx(start, offsets, granularity)
    map[start_idx[0]][start_idx[1]][start_idx[2]] = START_CHAR
    maze = Maze(map, offsets, granularity)
    return maze
