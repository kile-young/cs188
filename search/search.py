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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    print 'TYPES', type(s)

    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):

    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    return genericSearch(problem, 'dfs')





def DFSold():
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())


    # seen_Set = set()
    # stack_Fringe = [problem.getStartState()]

    # while stack_Fringe != []:
    #     node_state = stack_Fringe.pop()

    #     if problem.isGoalState(node_state[0:2]):
    #         goal_node = node_state

    #     if node_state[0:2] not in seen_Set:
    #         seen_Set.update({node_state[0:2]})

    #         for child in problem.getSuccessors(node_state[0:2]):
    #             stack_Fringe.append(child[0] + node_state)


    # current_node = goal_node
    # node_list = []

    # i = 0
    # while i < len(goal_node)-2:
    #     coord_next = (goal_node[i], goal_node[i+1])
    #     coord_last = (goal_node[i+2], goal_node[i+3])
    #     node_list.append(getDirection(coord_last, coord_next))
    #     i += 2

    # node_list = node_list[::-1]

    return []


    def currentDFS(problem):

        from util import Stack

        seen = set()
        stack = Stack()
        stack.push(problem.getStartState())
        prior = {}
        prior[problem.getStartState()] = None
        direcs = {}
        direcs[problem.getStartState()] = None

        while stack:
            current = stack.pop()

            if problem.isGoalState(current):
                goal_node = current
                break
            # seen.update({current})

            if current not in seen:
                seen.update({current})

                for child in problem.getSuccessors(current):
                    # WHY: for bfs it's checking prior and for dfs it's checking seen?
                    if child[0] not in seen:
                        stack.push(child[0])
                        prior[child[0]] = current
                        direcs[child[0]] = child[1]

        node_list = []
        # print direcs
        # print prior

        while goal_node != problem.getStartState():
            node_list.append(direcs[goal_node])
            goal_node = prior[goal_node]


        # print node_list[::-1]
        return node_list[::-1]
        # return []

        util.raiseNotDefined()


# def dfs(problem):
#     stack = [problem.getStartState()]
#     prior = {}
#     prior[problem.getStartState()] = None

#     while stack != []:
#         current = stack.pop()

#         if problem.isGoalState(current):
#             goal_node = current
#             break

#         for child in problem.getSuccessors(current):
#             if child[0] not in prior:
#                 stack.append(child[0])
#                 prior[child[0]] = current

#     node_list = []

#     while goal_node is not problem.getStartState():
#         # print getDirection(prior[goal_node], goal_node)
#         node_list.append(getDirection(prior[goal_node], goal_node))
#         goal_node = prior[goal_node]


#     print node_list[::-1]
#     return node_list[::-1]

#     util.raiseNotDefined()


def getDirection(coord_last, coord_next):

    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH

    # print 'coord_next: ', coord_next

    # print 'coord_next[0]: ', coord_next[0]

    if coord_last[0] == coord_next[0]:
        if coord_last[1] == coord_next[1]+1:
            return s
        else:
            return n
    elif coord_last[0] == coord_next[0]+1:
        return w
    else:
        return e

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    

    return genericSearch(problem, 'bfs')
    # from util import Queue

    # seen_Set = set()
    # queue_Fringe = deque([problem.getStartState()])

    # while len(queue_Fringe) != 0:
    #     node_state = queue_Fringe.popleft()
    #     print node_state
    #     if problem.isGoalState(node_state[0:2]):
    #         goal_node = node_state

    #     if node_state[0:2] not in seen_Set:
    #         seen_Set.update({node_state[0:2]})

    #         for child in problem.getSuccessors(node_state[0:2]):
    #             queue_Fringe.append(child[0] + node_state)


    # current_node = goal_node
    # node_list = []

    # i = 0
    # while i < len(goal_node)-2:
    #     coord_next = (goal_node[i], goal_node[i+1])
    #     coord_last = (goal_node[i+2], goal_node[i+3])
    #     node_list.append(getDirection(coord_last, coord_next))
    #     i += 2

    # node_list = node_list[::-1]

    # return node_list    
    # util.raiseNotDefined()

# ----------CURRENT------------ #

    # seen = set()
    # queue = Queue()
    # queue.push(problem.getStartState())
    # prior = {}
    # prior[problem.getStartState()] = None
    # direcs = {}
    # direcs[problem.getStartState()] = None

    # while queue:
    #     current = queue.pop()

    #     if problem.isGoalState(current):
    #         goal_node = current
    #         break

    #     if current not in seen:
    #             seen.update({current})

    #             for child in problem.getSuccessors(current):
    #                 # WHY: for bfs it's checking prior and for dfs it's checking seen?
    #                 if child[0] not in prior:
    #                     queue.push(child[0])
    #                     prior[child[0]] = current
    #                     direcs[child[0]] = child[1]

    # node_list = []

    # while goal_node != problem.getStartState():
    #     node_list.append(direcs[goal_node])
    #     goal_node = prior[goal_node]


    # # print node_list[::-1]
    # return node_list[::-1]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    

    return genericSearch(problem, 'ucs')
    # seen_Set = set()
    # queue_Fringe = PriorityQueue()
    # queue_Fringe.push(problem.getStartState(), 0)

    # while not queue_Fringe.isEmpty():
    #     node_state = queue_Fringe.pop()
    #     # print "Node Initial", node_state
    #     # node_state = node_state[0]
    #     # print "node state after", node_state
    #     # print 'iterations'
    #     if problem.isGoalState(node_state[0:2]):
    #         goal_node = node_state

    #     if node_state[0:2] not in seen_Set:
    #         seen_Set.update({node_state[0:2]})

    #         for child in problem.getSuccessors(node_state[0:2]):
    #             queue_Fringe.push(child[0] + node_state, child[2])


    # current_node = goal_node
    # node_list = []

    # i = 0
    # while i < len(goal_node)-2:
    #     coord_next = (goal_node[i], goal_node[i+1])
    #     coord_last = (goal_node[i+2], goal_node[i+3])
    #     # print getDirection(coord_last, coord_next)
    #     node_list.append(getDirection(coord_last, coord_next))
    #     i += 2

    # node_list = node_list[::-1]

    # print node_list
    # return node_list

# --------CURRENT------ #

    # seen = set()
    # fringe = PriorityQueue()
    # fringe.push(problem.getStartState(), 0)
    # prior = {}
    # prior[problem.getStartState()] = None
    # direcs = {}
    # direcs[problem.getStartState()] = None
    # path_cost = {}
    # path_cost[problem.getStartState()] = 0.0

    # while fringe != []:
    #     current = fringe.pop()

    #     if problem.isGoalState(current):
    #         goal_node = current
    #         break

    #     if current not in seen:
    #         seen.update({current})

    #         # print "Prior: ", prior
    #         # print "Seen: ", seen
    #         for child in problem.getSuccessors(current):
    #             # WHY: for bfs it's checking prior and for dfs it's checking seen?
    #             if child[0] not in prior:
    #                 path_cost[child[0]] = path_cost[current] + child[2]
    #                 fringe.push(child[0], path_cost[child[0]])
    #                 prior[child[0]] = current
    #                 direcs[child[0]] = child[1]

    # node_list = []

    # while goal_node != problem.getStartState():
    #     node_list.append(direcs[goal_node])
    #     goal_node = prior[goal_node]


    # print node_list[::-1]
    # return node_list[::-1]

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def genericSearch(problem, type, heuristic=nullHeuristic):

    from util import Stack, Queue, PriorityQueue

    path_cost = {}

    if type == 'dfs':
        fringe = Stack()
        fringe.push((problem.getStartState(), []))

    elif type == 'bfs':
        fringe = Queue()
        fringe.push((problem.getStartState(), []))

    elif type == 'ucs' or type =='astar':
        fringe = PriorityQueue()
        fringe.push((problem.getStartState(), []), 0)
        path_cost[problem.getStartState()] = 0.0

    seen = []
 

    while not fringe.isEmpty():


        current = fringe.pop()

        # print 'Current: ', current[0]

        if problem.isGoalState(current[0]):
            # print 'test2'
            break
      
        if current[0] not in seen:
            seen.append(current[0])

            # print 'Succs: ', problem.getSuccessors(current[0])

            for child in problem.getSuccessors(current[0]):
                # lst = current[1] + [child[1]]
                    if type == 'dfs' or type == 'bfs':
                        # print 'no?'
                        fringe.push((child[0], current[1]+[child[1]]))
                    elif type == 'ucs' or type == 'astar':
                        # print 'child: ', child[0], ' current: ', current[0]
                        path_cost[child[0]] = path_cost[current[0]] + child[2]
                        fringe.push((child[0], current[1]+[child[1]]), 
                            path_cost[child[0]] + heuristic(child[0], problem))

    # print 'Empty?: ', fringe.isEmpty()

    return current[1]

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    return genericSearch(problem, 'astar', heuristic)
    # from util import PriorityQueue

    # queue = PriorityQueue()
    # seen = set()
    # cum_cost = {}
    # prior = {}
    # cum_cost[problem.getStartState()] = 0
    # prior[problem.getStartState()] = None

    # queue.push(problem.getStartState(), 0) 

    # while not queue.isEmpty():
    #     current = queue.pop()

    #     if problem.isGoalState(current):
    #         goal_node = current
    #         break

    #     if current not in seen:
    #         seen.update({current})
    #         for child in problem.getSuccessors(current):
    #             new_cost = cum_cost[current] + child[2]
    #             # print child[0]
    #             # print 'new cost: ', new_cost
    #             if child[0] not in cum_cost or new_cost < cum_cost[child[0]]:
    #                 # print 'cum_cost_child', cum_cost[child[0]]
    #                 cum_cost[child[0]] = new_cost
    #                 priority = new_cost + heuristic(child[0], problem)
    #                 queue.push(child[0], priority)
    #                 # print 'Current: ', current
    #                 # print 'child: ', child[0]
    #                 prior[child[0]] = current
    #                 # print prior

    # # print goal_node
    # # print prior
    # node_list = []

    # while goal_node is not problem.getStartState():
    #     # print getDirection(prior[goal_node], goal_node)
    #     node_list.append(getDirection(prior[goal_node], goal_node))
    #     goal_node = prior[goal_node]


    # print node_list[::-1]
    # return node_list[::-1]

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
