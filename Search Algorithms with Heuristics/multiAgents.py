# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):


    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

 
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):


    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  

    def getAction(self, gameState):
        
        #Returns the minimax action from the current gameState using self.depth
        #and self.evaluationFunction.


        def max_agent(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()

            bestScore = float("-inf")
            score = bestScore
            bestAction = Directions.STOP

            for action in state.getLegalActions(0):
                score = exp_agent(state.generateSuccessor(0, action), depth, 1)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            if depth != 0:
                return bestScore
            else:
                return bestAction

        
        def exp_agent(state, depth, agent):
            if state.isLose() or state.isWin():
                return state.getScore()

            next = agent + 1

            if agent == state.getNumAgents() - 1:
                next = 0

            bestScore = float("inf")
            score = bestScore

            for action in state.getLegalActions(agent):
                if next != 0:
                    score = exp_agent(state.generateSuccessor(agent, action), depth, next)
                else:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(agent, action))
                    else:
                        score = max_agent(state.generateSuccessor(agent, action), depth + 1)
                
                if score < bestScore:
                    bestScore = score
            
            return bestScore
        
        return max_agent(gameState, 0)

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    

    def getAction(self, gameState):
        
        #Returns the expectimax action using self.depth and self.evaluationFunction

        def minmax(state, index, d, numAgents):
            if state.isWin() or state.isLose() or (d == 0):
                return (self.evaluationFunction(state), None) 
                
            if index == 1:
                max = -float("inf")
                maxState = None
                for child in state.getLegalActions(index-1):
                    if child != Directions.STOP:
                        successor = state.generateSuccessor(index-1, child)
                        x = minmax(successor,2,d,numagent)[0] 
                        if x > max:
                            max = x
                            maxState = child
                return (max, maxState)
            
            else: #min
                min = float("inf")
                minState = None
                avglist = []
                for child in state.getLegalActions(index-1):
                    if child != Directions.STOP:
                        successor = state.generateSuccessor(index-1, child)
                        if index != numAgents:
                            x = minmax(successor,index+1,d, numagent)[0]
                        else:
                            x = minmax(successor,1, d-1, numagent)[0]
                        avglist.append(x)
                        if x < min:
                            min = x
                            minState = child

                avg = float(sum(avglist)/len(avglist))
                return (avg, minState)
                

        numagent = gameState.getNumAgents()
        
        return minmax(gameState, 1, self.depth, numagent)[1]
        
        util.raiseNotDefined()


