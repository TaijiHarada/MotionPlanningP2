#!/usr/bin/env python
'''
Package providing implementation of a probabilistic roadmap algorithm
'''

import numpy as np
import matplotlib.pyplot as plotter
from collisions import PolygonEnvironment
import time
import heapq

_DEBUG = False
_DEBUG_END = True

def fake_in_collision(q):
    '''
    We never collide with this function!
    '''
    return False

def euclidean_heuristic(self, s, goal):
    '''
    Euclidean heuristic function

    s - configuration vector
    goal - goal vector

    returns - floating point estimate of the cost to the goal from state s
    '''
    return np.linalg.norm(s - goal)

class PriorityQ:
    '''
    Priority queue implementation with quick access for membership testing
    Setup currently to only with the SearchNode class
    '''
    def __init__(self):
        '''
        Initialize an empty priority queue
        '''
        self.l = [] # list storing the priority q
        self.s = set() # set for fast membership testing

    def __contains__(self, x):
        '''
        Test if x is in the queue
        '''
        return x in self.s

    def push(self, x, cost):
        '''
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        '''
        if tuple(x.state.tolist()) in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(tuple(x.state.tolist()))

    def pop(self):
        '''
        Get the value and remove the lowest cost element from the queue
        '''
        x = heapq.heappop(self.l)
        self.s.remove(tuple(x[1].state.tolist()))
        return x[1]

    def peak(self):
        '''
        Get the value of the lowest cost element in the priority queue
        '''
        x = self.l[0]
        return x[1]

    def __len__(self):
        '''
        Return the number of elements in the queue
        '''
        return len(self.l)

    def replace(self, x, new_cost):
        '''
        Removes element x from the q and replaces it with x with the new_cost
        '''
        for y in self.l:
            if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
                self.l.remove(y)
                self.s.remove(tuple(y[1].state.tolist()))
                break
        heapq.heapify(self.l)
        self.push(x, new_cost)

    def get_cost(self, x):
        for y in self.l:
            if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
                return y[0]

    def __str__(self):
        return str(self.l)

def backpath(node):
    '''
    Function to determine the path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    '''
    path = []
    while node.parent is not None:
        path.append(node.state)
        node = node.parent
    path.append(node.state)
    path.reverse()
    return path

class StraightLinePlanner:
    def __init__(self, step_size, collision_func = None):
        self.in_collision = collision_func
        self.epsilon = step_size
        if collision_func is None:
            self.in_collision = fake_in_collision

    def plan(self, start, goal):
        '''
        Check if edge is collision free, taking epsilon steps towards the goal
        Returns: None / False if edge in collsion
                 Plan / True if edge if free
        '''
        raise NotImplementedError('Check if straight line edge between neighbours are collision free')

class RoadMapNode:
    '''
    Nodes to be used in a built RoadMap class
    '''
    def __init__(self, state, cost=0, parent=None):
        self.state = np.array(state)
        self.neighbors = []
        self.cost = 0
        self.parent = parent

    def add_neighbor(self, n_new):
        '''
        n_new - new neighbor
        '''
        self.neighbors.append(n_new)

    def is_neighbor(self, n_test):
        '''
        Test if n_test is already our neighbor
        '''
        for n in self.neighbors:
            if np.linalg.norm(n.state - n_test.state) == 0.0:
                return True
        return False

    def __eq__(self, other):
        return np.linalg.norm(self.state - other.state) == 0.0

class RoadMap:
    '''
    Class to store a built roadmap for searching in our multi-query PRM
    '''
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node, neighbors):
        '''
        Add a node to the roadmap. Connect it to its neighbors
        '''
        # Avoid adding duplicates
        self.nodes.append(node)
        for n in neighbors:
            node.add_neighbor(n)
            if not n.is_neighbor(node):
                n.add_neighbor(node)
                self.edges.append((n.state, node.state))
                
    def get_states_and_edges(self):
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

class PRM:
    def __init__(self, num_samples, local_planner, num_dimensions, lims = None,
                 collision_func = None, radius=2.0, epsilon=0.1):
        self.local_planner = local_planner
        self.r = radius
        self.N = num_samples
        self.n = num_dimensions
        self.epsilon = epsilon

        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = fake_in_collision

        # Setup range limits
        self.limits = lims
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0,100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:,1] - self.limits[:,0]

        # Build the roadmap instance
        self.T = RoadMap()

    def build_prm(self, reset=False):
        '''
        reset - empty the current roadmap if requested
        '''
        if reset:
            self.T = RoadMap()

        raise NotImplementedError('Sample configurations and build a roadmap')

    def find_valid_neighbors(self, n_query, samples, r):
        '''
        Find the nodes that are close to n_query and can be attached by the local planner
        returns - list of neighbors reached by the local planner
        '''
        valid_neighbors = []
        raise NotImplementedError('Find samples withing radius r of n_query that is collision free')
        return valid_neighbors
    
    def query(self, start, goal):
        '''
        Generate a path from start to goal using the built roadmap
        returns - Path of configurations if in roadmap, None otherwise
        '''
        start_node = RoadMapNode(start)
        goal_node = RoadMapNode(goal)

        raise NotImplementedError('Attach start and goal node to the roadmap self.T')

        def is_goal(x):
            '''
            Test if a sample is at the goal
            '''
            return np.linalg.norm(x - goal) < self.epsilon

        # Run search on the roadmap to find a plan
        start_node.parent = None
        plan, visited = self.graph_search(start_node, is_goal)
        return plan, visited


    def uniform_cost_search(self, init_node, is_goal):
        '''
        Perform graph search on the roadmap
        '''
        cost = 0
        frontier = PriorityQ()
        frontier.push(init_node, cost)
        visited = set()
        #You need to modify your graph search from HW1 to expand neighbors instead of actions
        raise NotImplementedError('Add in your favorite optimal graph search from HW1')
        return None, visited
    
    def sample(self):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''
        raise NotImplementedError('Sample a new configuration')


def saveFig(name,close = True):
    plotter.savefig(name + ".png")
    if(close):
        plotter.close()

  
def test_prm_env(num_samples=50, step_length=1, env='./env0.txt'):
    pe = PolygonEnvironment()
    pe.read_env(env)

    dims = len(pe.start)
    start_time = time.time()

    local_planner = StraightLinePlanner(step_length, pe.test_collisions)
    prm = PRM(num_samples,
              local_planner,
              dims,
              radius = 20,
              epsilon = step_length,
              lims = pe.lims,
              collision_func=pe.test_collisions)
    print('Builing PRM')
    prm.build_prm()
    build_time = time.time() - start_time
    print('Build time', build_time)
    print('Finding Plan')
    plan, visited = prm.query(pe.start, pe.goal)
    pe.draw_env(show=False)
    pe.draw_plan(None, prm,False,True,True)

    run_time = time.time() - start_time
    print('plan:', plan)
    print('run_time =', run_time)


    return plan, prm, visited
if __name__== "__main__":
  test_prm_env()
