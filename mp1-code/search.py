# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import queue
from queue import PriorityQueue

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    start = maze.getStart()
    frontier = queue.Queue()
    frontier.put(start)
    end = ()
    previous = {}

    while(not frontier.empty()):
        current = frontier.get()
        if maze.isObjective(current[0], current[1]):
            end= (current[0], current[1])
            break

        neighbors = maze.getNeighbors(current[0], current[1])
        for n in neighbors:
            if (n not in previous):
                frontier.put(n)
                previous[n] = current

    return backtrace(previous, start, end)

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontier = PriorityQueue()
    start = maze.getStart()
    goal = (maze.getObjectives())[0]
    end = ()
    cost = {}
    previous = {}
    frontier.put((0,start))
    previous[start] = None
    cost[start] = 0

    while(not frontier.empty()):
        current = frontier.get()[1]
        if maze.isObjective(current[0], current[1]):
            end= (current[0], current[1])
            break

        g = cost[current]+ 1
        neighbors = maze.getNeighbors(current[0], current[1])
        for n in neighbors:
            if (n not in previous) or (g < cost[n]):
                if(n not in previous):
                    h = heuristic(current, goal)
                    f = g + h
                    frontier.put((f, n))
                cost[n] = g
                previous[n] = current

    return backtrace(previous, start, end)

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_multi(maze)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontier = PriorityQueue()
    goals = maze.getObjectives()
    remainder_goals = set(goals)
    graph = build_graph(maze)
    length_mst = build_mst(graph, remainder_goals)[1]
    start = (maze.getStart(), tuple(goals))
    end = ()
    cost = {}
    previous = {}
    path = []
    frontier.put((0,start))
    time = 0
    previous[start] = None
    cost[start] = 0
    print(length_mst)
    while(not frontier.empty()):
        current = frontier.get(previous)[1]
        remainder_goals = set(current[1])
        if current[0] in remainder_goals:
            remainder_goals.remove(current[0])
            if not remainder_goals or time == 3000:
                end = current
                break
            length_mst = build_mst(graph, remainder_goals)[1]

        g = cost[current]+ 1
        neighbors = maze.getNeighbors(current[0][0], current[0][1])
        for n in neighbors:
            if (n, tuple(remainder_goals)) not in previous or (g < cost[(n, tuple(remainder_goals))]):
                if (n, tuple(remainder_goals)) not in previous:
                    h = nearest_distance(maze, n, remainder_goals) + length_mst
                    f = g + h
                    frontier.put((f, (n, tuple(remainder_goals))))
                cost[(n, tuple(remainder_goals))] = g
                previous[(n, tuple(remainder_goals))] = current
        time +=1
    backtrace_multi(previous, path, start, end)
    print(path)
    return path

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []

def heuristic (current, goal):
    h = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    return h

def backtrace(previous, start, end):
    path = []
    while end is not start:
        path.append(end)
        end = previous[end]
    path.append(start)
    path.reverse()
    return path

def backtrace_multi(previous, path, start, end):
    while end is not start:
        path.append(end[0])
        end = previous[end]
    path.append(start[0])
    path.reverse()
    return path

def build_graph(maze):
    graph = {}
    goals = maze.getObjectives()
    for g in goals:
        graph[g] = {}
        for i in goals:
            if g is not i:
                graph[g][i] = len(astar_single(maze, g, i)) -1
    return graph

def build_mst(graph, goals):
    mst = []
    visit = goals.copy()
    keys = {}
    length_mst = 0
    for goal in visit:
        if not keys:
            keys[goal] = 0
        else:
            keys[goal] = float("inf")
    while visit:
        minimum_key = visit.pop()
        visit.add(minimum_key)
        for v in visit:
            if keys[v] < keys[minimum_key]:
                minimum_key = v
        mst.append(minimum_key)
        visit.remove(minimum_key)
        length_mst += keys[minimum_key]
        for neighbor in visit:
            distance = graph[minimum_key][neighbor]
            if distance < keys[neighbor]:
                keys[neighbor] = distance
    return (mst, length_mst)

def astar_single(maze, start, end):
    frontier = PriorityQueue()
    cost = {}
    previous = {}
    frontier.put((0,start))
    previous[start] = None
    cost[start] = 0

    while(not frontier.empty()):
        current = frontier.get()[1]
        if current == end:
            break

        g = cost[current]+ 1
        neighbors = maze.getNeighbors(current[0], current[1])
        for n in neighbors:
            if (n not in previous) or (g < cost[n]):
                if(n not in previous):
                    h = heuristic(current, end)
                    f = g + h
                    frontier.put((f, n))
                cost[n] = g
                previous[n] = current

    return backtrace(previous, start, end)

def nearest_distance(maze, current, goals):
    goals = set(goals)
    nearest = goals.pop()
    goals.add(nearest)
    distance = heuristic(current, nearest)
    for g in goals:
        if (heuristic(current, g) < distance):
            distance = heuristic(current, g)
    return distance
