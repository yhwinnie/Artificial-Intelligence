# policyIterationAgents.py
# ------------------------
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
import numpy as np

from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 20):
        """
          Your policy iteration agent should take an mdp on
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
        states = self.mdp.getStates()
        # initialize policy arbitrarily
        self.policy = {}
        for state in states:
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
                self.policy[state] = self.mdp.getPossibleActions(state)[0]
        # initialize policyValues dict

        self.policyValues = {}
        for state in states:
            self.policyValues[state] = 0

        for i in range(self.iterations):
            # step 1: call policy evaluation to get state values under policy, updating self.policyValues
            self.runPolicyEvaluation()
            # step 2: call policy improvement, which updates self.policy
            self.runPolicyImprovement()

    def runPolicyEvaluation(self):
        """ Run policy evaluation to get the state values under self.policy. Should update self.policyValues.
        Implement this by solving a linear system of equations using numpy. """

        states = []
        for key in self.policy:
            states += [key]

        A = np.eye(len(states))
        b = np.zeros(len(states))

        for state in states:
            if not self.mdp.isTerminal(state):
                i = 0 
                for iter_state in states:
                    if iter_state == state:
                        break
                    i += 1
                action = self.getPolicy(state)
                for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    k = 0
                    for iter_state in states:
                        if iter_state == new_state:
                            break
                        k += 1
                    if i != k:
                        A[i][k] = -1 * (self.discount * prob) 
                    else:
                        A[i][k] = 1 - (self.discount * prob)  

        for state in states:
            if not self.mdp.isTerminal(states):
                total_prob = 0
                i = 0
                for iter_state in states:
                    if iter_state == state:
                        break
                    i += 1
                for action in self.mdp.getPossibleActions(state):
                    for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        total_prob += prob 
                    b[i] = self.mdp.getReward(state) * total_prob

        x = np.linalg.solve(A, b)

        for i in range(len(x)):
            self.policyValues[states[i]] = x[i]

    def runPolicyImprovement(self):
        """ Run policy improvement using self.policyValues. Should update self.policy. """
        states = self.mdp.getStates()
        for state in states:
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
                highest_q = float('-inf')
                best_action = None
                for action in self.mdp.getPossibleActions(state):
                    temp_q = self.computeQValueFromValues(state, action)
                    if temp_q > highest_q:
                        highest_q = temp_q
                        best_action = action
                self.policy[state] = best_action

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.policyValues.
        """
        Qvalue = 0

        for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          Qvalue += prob * (self.mdp.getReward(state) + (self.discount * self.policyValues[new_state]))
        return Qvalue
        

    def getValue(self, state):
        return self.policyValues[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
