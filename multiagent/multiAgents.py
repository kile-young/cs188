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
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        value = successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        
        ghost_distances = []
        for ghost in newGhostStates:
          if ghost.scaredTimer == 0:
            if manhattanDistance(newPos, ghost.getPosition()) != 0:
              ghost_distances.append(manhattanDistance(newPos, ghost.getPosition()))



        food_distances = []
        for food in newFood.asList():
          if manhattanDistance(newPos, food) != 0:
            food_distances.append(manhattanDistance(newPos, food))
        

        ghost_distances.sort()

        food_distances.sort()

        if len(ghost_distances) > 0:
          value += -10.0/ghost_distances[len(ghost_distances)-1]
        if len(food_distances) > 0:
          value += 10.0/food_distances[0]
        
        return value

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
      action = self.value(gameState, 0)
      return action

    def value(self, state, depth):
      if state.isWin() or state.isLose() or depth == self.depth*state.getNumAgents():
        return self.evaluationFunction(state)
      if depth%state.getNumAgents() != 0:
        return self.minValue(state, depth)
      else: 
        return self.maxValue(state, depth)
      

    def minValue(self, state, depth):
      minimum = pow(2,32)

      actions = state.getLegalActions(depth%state.getNumAgents())

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        successor = state.generateSuccessor(depth%state.getNumAgents(), act)
        temp = self.value(successor, depth+1)
        if temp < minimum:
          minimum = temp
      return minimum

    def maxValue(self, state, depth):
      maximum = -pow(2,32)

      actions = state.getLegalActions(0)

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        successor = state.generateSuccessor(depth%state.getNumAgents(), act)
        temp = self.value(successor, depth+1)
        if temp > maximum:
          maximum = temp
          maxAct = act
      if depth == 0:
        return maxAct
      else:
        return maximum
      

        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.value(gameState, 0, -pow(2, 32), pow(2, 32))
        return action

    def value(self, state, depth, a, b):
      if state.isWin() or state.isLose() or depth ==  self.depth * state.getNumAgents():
        return self.evaluationFunction(state)
      if depth%state.getNumAgents() != 0:
        return self.minValue(state, depth, a, b)
      else: 
        return self.maxValue(state, depth, a, b)


    def minValue(self, state, depth, a, b):
      minimum = pow(2,32)

      actions = state.getLegalActions(depth%state.getNumAgents())

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        successor = state.generateSuccessor(depth%state.getNumAgents(), act)
        temp = self.value(successor, depth+1, a, b)
        if temp < minimum:
          minimum = temp
        if minimum < a:
          return minimum
        b = min(b, minimum)
      return minimum

    def maxValue(self, state, depth, a, b):
      maximum = -pow(2, 32)

      actions = state.getLegalActions(0)

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        successor = state.generateSuccessor(0, act)
        temp = self.value(successor, depth+1, a, b)
        if temp > maximum:
          maximum = temp
          maxAct = act
        if maximum > b:
          if depth == depth%state.getNumAgents():
            return maxAct
          else:
            return maximum
        a = max(a, maximum)
      if depth == depth%state.getNumAgents():
        return maxAct
      else:
        return maximum







class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = self.value(gameState, 0)
        return action

    def value(self, state, depth):

      if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
        return self.evaluationFunction(state)
      elif depth%state.getNumAgents() != 0:
        return self.expValue(state, depth)
      else:
        return self.maxValue(state, depth)

    def maxValue(self, state, depth):
      val = -pow(2,32)
      actions = state.getLegalActions(0)

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        successor = state.generateSuccessor(0, act)
        temp = self.value(successor, depth+1)
        if temp > val:
          val = temp
          maxAct = act

      if depth == 0:
        return maxAct
      else:
        return val


    def expValue(self, state, depth):
      val = 0
      actions = state.getLegalActions(depth%state.getNumAgents())

      if actions == []:
        return self.evaluationFunction(state)

      for act in actions:
        p = 1.0 / len(actions)
        successor = state.generateSuccessor(depth%state.getNumAgents(), act)
        val += p * self.value(successor, depth+1)
      return val

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    current = currentGameState.getPacmanPosition()
    value = currentGameState.getScore()
    # print 'Value: ', value
    # print type(value)

    maxFood = 0.0
    for food in currentGameState.getFood().asList():
      distance = manhattanDistance(current, food)
      if 5.0 / distance > maxFood:
        maxFood = 5.0 / distance

    value += maxFood

    x=0
    all_ghosts = currentGameState.getGhostStates()
    for ghost in all_ghosts:
      distance = manhattanDistance(current, ghost.getPosition())
      if distance > 0:
        if ghost.scaredTimer < 0:
          x += -100.0 / distance
        else:
          x += 1000.0 / distance

    value += x
    
    return value

# Abbreviation
better = betterEvaluationFunction

