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
import time

class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()

        max_val = float("-inf")
        i = 0
        for iteration in range(self.iterations):
          if (i % len(states)) < len(states):
            if self.mdp.isTerminal(states[(i % len(states))]):
              self.values[states[(i % len(states))]] = 0
              i += 1
            else:
              max_val = float("-inf")
              for action in self.mdp.getPossibleActions(states[(i % len(states))]):
                val = 0.0
                for new_state, prob in self.mdp.getTransitionStatesAndProbs(states[(i % len(states))], action):
                  val += (prob * (self.mdp.getReward(states[(i % len(states))]) + (self.discount * self.values[new_state])))
                if val > max_val:
                  max_val = val
              self.values[states[(i % len(states))]] = max_val
              i += 1

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
        Qvalue = 0

        for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          # if self.getValue(new_state) is not None:
          Qvalue += prob * (self.mdp.getReward(state) + (self.discount * self.getValue(new_state)))
        return Qvalue

          

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Policy 
        if self.mdp.isTerminal(state):
          return None
        else:

          best_policy = float("-inf")
          best_action = None
          for action in self.mdp.getPossibleActions(state):
            temp_policy = self.computeQValueFromValues(state, action)

            if temp_policy > best_policy:
              best_policy = temp_policy
              best_action = action 
        return best_action 


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


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

        import util
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        prior_q = util.PriorityQueue()
        pred_dict = {}
        for state in states:
          predecessors = set()
          for parent in states:
            for action in self.mdp.getPossibleActions(parent):
              for pred_state, prob in self.mdp.getTransitionStatesAndProbs(parent, action):
                if not self.mdp.isTerminal(parent):
                  if prob != 0.0:
                    predecessors.add(parent)
          pred_dict[state] = predecessors


          if not self.mdp.isTerminal(state):
            highest_q = float("-inf")
            for action in self.mdp.getPossibleActions(state):
              q_val = self.computeQValueFromValues(state, action)

              if q_val > highest_q:
                highest_q = q_val

            diff = abs(self.values[state] - highest_q)
            prior_q.update(state, -diff)


        for i in range(self.iterations):
          if prior_q.isEmpty():
            return None
          else: 
            state = prior_q.pop()
            predecessors = pred_dict[state]
            
            if not self.mdp.isTerminal(state):
              highest_q = float("-inf")
              for action in self.mdp.getPossibleActions(state):
                q_val = 0.0

                for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                  q_val += prob * (self.mdp.getReward(state) + (self.discount * self.values[new_state]))
                if q_val > highest_q:
                  highest_q = q_val

              self.values[state] = highest_q

            for p in predecessors:
              max_q = float("-inf")
              for action in self.mdp.getPossibleActions(p):
                temp_q = self.computeQValueFromValues(p, action)

                if temp_q > max_q:
                  max_q = temp_q

              diff = abs(self.values[p] - max_q)
              if diff > theta:
                prior_q.update(p, -diff)

        




