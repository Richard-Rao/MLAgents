# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = {}

    def getQValue(self, state, action):
      
          #Returns Q(state,action) or 0.0 if not seen
          
        if (state,action) in self.qvalues:
          return self.qvalues[(state,action)]
        else:
          return 0.0


    def computeValueFromQValues(self, state):
        
        #Returns max_action Q(state,action)
        
        qvals = []
        for action in self.getLegalActions(state):
          qvals.append(self.getQValue(state,action))

        if len(qvals):
          return max(qvals)
        else:
          return 0.0


    def computeActionFromQValues(self, state):
        #returns best action or None if no actions
        bestVal = self.getValue(state)
        bestActions = []
        for action in self.getLegalActions(state):
          if self.getQValue(state, action) == bestVal:
            bestActions.append(action)

        if len(bestActions):
          return random.choice(bestActions)
        else:
          return None



    def getAction(self, state):
        #chooses best action from legal actions or None
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        alpha = self.alpha
        discount = self.discount
        qval = self.getQValue(state, action)
        next_value = self.getValue(nextState)
        
        newVal = (1-alpha) * qval + alpha * (reward + discount * next_value)

        self.qvalues[(state, action)] = newVal

        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        #Informs the parent of which actions is taken
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        
          #returns Q(state,action) = w * featureVector where * is the dotProduct operator
        
        featVec = self.featExtractor.getFeatures(state,action)
        return self.weights * featVec

    def update(self, state, action, nextState, reward):
        #updates weights based on transition
        featVec = self.featExtractor.getFeatures(state,action)
        prevVal = self.getQValue(state,action)
        nextVal = self.getValue(nextState)
        diff = (reward + self.discount * nextVal) - prevVal

        for feature in featVec:
          newWeight = self.alpha * diff *  featVec[feature]
          self.weights[feature] += newWeight


    def final(self, state):
        
        PacmanQAgent.final(self, state)
