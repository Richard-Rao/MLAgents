# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
          states = self.mdp.getStates()
          counter = util.Counter()
          
          for state in states:
            maxVal = float("-inf")
            
            for action in self.mdp.getPossibleActions(state):
              qval = self.computeQValueFromValues(state, action)
              
              if qval > maxVal:
                maxVal = qval
              
              counter[state] = maxVal
          
          self.values = counter


    def getValue(self, state):
        #returns value of state
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        #computes Q-value of action in state using Values
        actionPairs = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        
        for next_state, prob in actionPairs:
            reward = self.mdp.getReward(state, action, next_state)
            total += prob * (reward + self.discount * self.values[next_state])
        
        return total

    def computeActionFromValues(self, state):
        #determines which action will result in highest value(best move)
        bestAction = None
        maxVal = float("-inf")
        for action in self.mdp.getPossibleActions(state):
          qval = self.computeQValueFromValues(state, action)
          
          if qval > maxVal:
            maxVal = qval
            bestAction = action
        
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

