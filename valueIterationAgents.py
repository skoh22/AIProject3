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
import random
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        utilityPrime = util.Counter()
        states = mdp.getStates()
        count = 0
        while count < iterations:  # or delta < 1.0 * epsilon * (1 - discount) / discount or
            self.values = utilityPrime.copy()
            #print 'COUNT', count
            #delta = 0
            # looping through states
            for i in range(len(states)):  # states[0] is terminal, throws off values bc no actions -> gets assigned -99999
                state = states[i]
                if i is 0:
                    maxProbUtil = 0
                    #maxAction = None
                else:
                    #print 'STATE:', state
                    actions = mdp.getPossibleActions(state)
                    #print 'ACTIONS:', actions
                    maxProbUtil = -99999
                    #maxAction = None
                    for action in actions:
                        results = mdp.getTransitionStatesAndProbs(state, action)
                        probUtil = sum(prob * self.values[nextState] for nextState, prob in results)
                        if probUtil > maxProbUtil:
                            maxProbUtil = probUtil
                            #maxAction = action
                    #print 'MAX UTIL:', maxProbUtil
                    # why does reward depend on next state
                utilityPrime[state] = mdp.getReward(state, None, None) + discount * maxProbUtil
                #if abs(utilityPrime[state] - self.values[state]) > delta:
                    #delta = abs(utilityPrime[state] - self.values[state])
            count = count + 1  # was indented one step too far -> iterating 12x less than should have been
        self.values = utilityPrime.copy()
            # print self.values
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        results = self.mdp.getTransitionStatesAndProbs(state,action)
        return sum(prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState]) for nextState, prob in results)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        else:
            maxVal = -999999
            bestActionList = []
            for action in actions:
                val = self.getQValue(state,action)
                if val>maxVal:
                    #ties broken by first action listed in getpossibleactions
                    bestActionList = [action]
                    maxVal = val
                elif val == maxVal:
                    bestActionList.append(action)
            #print "bestActionList: ", bestActionList
            return random.choice(bestActionList)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
