# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).




import util
import sys
import copy

class SearchProblem:
    

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    
    #Performs DFS but every iteration, increases the max depth by 1
    depth = 0
    limit = 1000
    for depth in range(limit):
        result = depthLimitedSearch(problem, depth)
        if result != "tooDeep":
            if result:
                return result
        depth+=1

def depthLimitedSearch(problem, limit=1000):
    
    startingState = problem.getStartState()
    if problem.goalTest(startingState):
        return []
    
    visitedStates = []
    tooDeep = False
    q = util.Queue()
    q.push(Node(startingState, None, [], 0))

    while not q.isEmpty():
        if tooDeep: 
            return "tooDeep"
        
        node = q.pop()

        currentState = node.state
        actions = node.action
        prevCost = node.path_cost

        if prevCost == limit: 
            tooDeep = True
        else:
            if currentState not in visitedStates:
                visitedStates.append(currentState)

                if problem.goalTest(currentState):
                    return actions

                for move in problem.getActions(currentState):
                    nextState = problem.getResult(currentState,move)
                    newAction = actions + [move]
                    newCost = prevCost + problem.getCost(currentState, move)
                    q.push(Node(nextState, node, newAction, newCost))



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    startingState = problem.getStartState()
    if problem.goalTest(startingState):
        return []
    
    visitedNodes = []

    prioQ =util.PriorityQueue()
    prioQ.push(Node(startingState, None, [], 0), 0)

    while not prioQ.isEmpty():
        node = prioQ.pop()
        currentState = node.state
        actions = node.action
        prevCost = node.path_cost

        if currentState not in visitedNodes:
            visitedNodes.append(currentState)

            if problem.goalTest(currentState):
                return actions

            for move in problem.getActions(currentState):
                nextState = problem.getResult(currentState,move)
                newAction = actions + [move]
                newCost = prevCost + problem.getCost(currentState, move)
                heuristicCost = newCost + heuristic(nextState, problem)
                prioQ.push(Node(nextState, node, newAction, newCost),heuristicCost)

    util.raiseNotDefined()

#Abbreviations
astar = aStarSearch
ids = iterativeDeepeningSearch
