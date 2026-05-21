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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()

        if len(foodList) > 0:
            minFoodDist = min(
                [manhattanDistance(newPos, food) for food in foodList]
            )

            score += 1.0 / (minFoodDist + 1)

        for ghostState in newGhostStates:

            ghostPos = ghostState.getPosition()

            dist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:

                if dist <= 1:
                    score -= 1000

                else:
                    score -= 2.0 / (dist + 1)
                

            # Ghost scared
            else:
                score += 2.0 / (dist + 1)


        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def minimax(self, gameState: GameState, depth: int, agent: int):
        if depth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agent) == 0:
            return self.evaluationFunction(gameState)
        
        # pacman -> ghost 1 -> ghost 2 -> ... -> ghost n -> pacman -> ...
        next_agent = (agent + 1) % gameState.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth
        ret_value = float("-inf") if agent == 0 else float("inf")

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            if agent == 0:
                ret_value = max(ret_value, self.minimax(next_state, next_depth, next_agent))
            else:
                ret_value = min(ret_value, self.minimax(next_state, next_depth, next_agent))

        return ret_value

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"

        best_score = float("-inf")
        best_action = None
        pacman = 0
        depth = 0
        for action in gameState.getLegalActions(pacman):
            next_state = gameState.generateSuccessor(pacman, action)
            next_value = self.minimax(next_state, depth, pacman + 1)
            if next_value > best_score:
                best_score = next_value
                best_action = action
        
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (q3)
    """

    def alphabeta(self, gameState: GameState, depth: int, agent: int, alpha: float, beta: float):
        # Terminal check
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agent)
        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)

        next_agent = (agent + 1) % gameState.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        if agent == 0:
            # MAX node (Pacman)
            value = float("-inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)
                value = max(value, self.alphabeta(successor, next_depth, next_agent, alpha, beta))
                # không prune khi bằng nhau (yêu cầu của autograder)
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        else:
            # MIN node (Ghost)
            value = float("inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.alphabeta(successor, next_depth, next_agent, alpha, beta))
                # không prune khi bằng nhau (yêu cầu của autograder)
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        best_action = None
        best_score = float("-inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.alphabeta(successor, 0, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            # Cập nhật alpha tại root
            alpha = max(alpha, best_score)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState: GameState, depth: int, agent: int):
        if (
            depth == self.depth
            or gameState.isWin()
            or gameState.isLose()
        ):
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agent)

        if len(legalActions) == 0:
            return self.evaluationFunction(gameState)

        next_agent = (agent + 1) % gameState.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        if agent == 0:
            value = float("-inf")

            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)
                value = max(
                    value,
                    self.expectimax(successor, next_depth, next_agent)
                )

            return value
        else:
            probability = 1.0 / len(legalActions)
            expected_value = 0

            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)
                expected_value += probability * self.expectimax(
                    successor,
                    next_depth,
                    next_agent
                )

            return expected_value

    def getAction(self, gameState: GameState):
        best_score = float("-inf")
        best_action = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            score = self.expectimax(
                successor,
                0,
                1
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if len(food) > 0:
        minFoodDist = min(manhattanDistance(pos, f) for f in food)
        score += 10.0 / minFoodDist

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        if dist > 0:
            if ghost.scaredTimer > 0:
                score += 25.0 / dist
            else:
                score -= 10.0 / dist
        else:
            """
            Khi pacman bị ma bắt
            """
            score -= 500.0
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction