#!/usr/bin/env python
'''
Package providing helper classes and functions for performing graph search operations for planning.
'''
import numpy as np
import matplotlib.pyplot as plotter
from math import pi
from collisions import PolygonEnvironment
import time
import random

_DEBUG = True

_TRAPPED = 'trapped'
_ADVANCED = 'advanced'
_REACHED = 'reached'

class TreeNode:
    '''
    Class to hold node state and connectivity for building an RRT
    '''
    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child):
        '''
        Add a child node
        '''
        self.children.append(child)

class RRTSearchTree:
    '''
    Searh tree used for building an RRT
    '''
    def __init__(self, init):
        '''
        init - initial tree configuration
        '''
        self.root = TreeNode(init)
        self.nodes = [self.root]
        self.edges = []

    def find_nearest(self, s_query):
        '''
        Find node in tree closets to s_query
        returns - (nearest node, dist to nearest node)
        '''
        min_d = 1000000
        nn = self.root
        for n_i in self.nodes:
            d = np.linalg.norm(s_query - n_i.state)
            if d < min_d:
                nn = n_i
                min_d = d
        return (nn, min_d)

    def add_node(self, node, parent):
        '''
        Add a node to the tree
        node - new node to add
        parent - nodes parent, already in the tree
        '''
        self.nodes.append(node)
        self.edges.append((parent.state, node.state))
        node.parent = parent
        parent.add_child(node)

    def get_states_and_edges(self):
        '''
        Return a list of states and edgs in the tree
        '''
        states = np.array([n.state for n in self.nodes])
        return (states, self.edges)

    def get_back_path(self, n):
        '''
        Get the path from the root to a specific node in the tree
        n - node in tree to get path to
        '''
        path = []
        while n.parent is not None:
            path.append(n.state)
            n = n.parent
        path.append(n.state)
        path.reverse()
        return path

class RRT(object):
    '''
    Rapidly-Exploring Random Tree Planner
    '''
    def __init__(self, num_samples, num_dimensions=2, step_length = 1, lims = None,
                 connect_prob = 0.05, collision_func=None):
        '''
        Initialize an RRT planning instance
        '''
        # Samples to generate
        self.K = num_samples
        
        self.n = num_dimensions
        
        self.epsilon = step_length
        '''
        Bias Towards Goal
        '''
        self.connect_prob = connect_prob

        self.in_collision = collision_func
        if collision_func is None:
            self.in_collision = self.fake_in_collision

        # Setup range limits
        self.limits = lims
        # If limits are not provided, provide defaults.
        if self.limits is None:
            self.limits = []
            for n in range(num_dimensions):
                self.limits.append([0,100])
            self.limits = np.array(self.limits)

        self.ranges = self.limits[:,1] - self.limits[:,0]
        self.found_path = False

    def build_rrt(self, init, goal):
        '''
        Build the RRT from init to goal.
        Returns a path to goal or None if no path is found.
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        # Loop through the number of samples
        if _DEBUG:
            print('Test Sample:', self.sample())
        for _ in range(self.K):
            # Sample a random point
            q_rand = self.sample()  
            # Get Status of extend(reached, advanced, or trapped)
            status, new_node = self.extend(self.T, q_rand)

            if status == _REACHED and np.linalg.norm(new_node.state - self.goal) < self.epsilon:
                self.found_path = True
                return self.T.get_back_path(new_node)  # Return final path

        return None  # No path found

    def build_rrt_connect(self, init, goal):
        '''
        Build the rrt connect from init to goal
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build tree and search
        self.T = RRTSearchTree(init)

        # Sample and extend
        raise NotImplementedError('Expand RRT tree and return plan')

        return None

    def build_bidirectional_rrt_connect(self, init, goal):
        '''
        Build two rrt connect trees from init and goal
        Growing towards each oter
        Returns path to goal or None
        '''
        self.goal = np.array(goal)
        self.init = np.array(init)
        self.found_path = False

        # Build trees and search
        self.T_init = RRTSearchTree(init)
        self.T_goal = RRTSearchTree(goal)

        # Sample and extend
        raise NotImplementedError('Expand RRT trees and return plan')

        return None

    def sample(self):
        '''
        Sample a new configuration
        Returns a configuration of size self.n bounded in self.limits
        '''
        # Return goal with a connect_prob probability.  I.e if connect_prob is 5 %, return goal 5% of the time.
        if random.random() < self.connect_prob:
            return self.goal  # Bias towards goal

        # Return a sample within the limits of each dimension.
        samples = []
        for i in range(self.n):
            lower_bound = self.limits[i, 0]
            upper_bound = self.limits[i, 1]
            sample = random.uniform(lower_bound, upper_bound)
            samples.append(sample)
        return np.array(samples)

    def extend(self, T, q):
        '''
        Perform rrt extend operation.
        q - new configuration to extend towards
        returns - tuple of (status, TreeNode)
           status can be: _TRAPPED, _ADVANCED or _REACHED

        Self notes: selects the nearest vertex already in the
        RRT to the given sample configuration
        '''
        # Kuffner psuodocode
        #         EXTEND(T , q)
        # 1 qnear ← NEAREST NEIGHBOR(q, T );
        # 2 if NEW CONFIG(q, qnear , qnew ) then
            # 3 T .add vertex(qnew );
            # 4 T .add edge(qnear , qnew );
            # 5 if qnew = q then
            # 6 Return Reached;
        # 7 else
            # 8 Return Advanced;
        # 9 Return Trapped;

        # Kuffner statuses Three situations can occur:
        # Reached, in which q is directly added to the RRT be-
        # cause it already contains a vertex within eplilon of q; Ad-
        # vanced, in which a new vertex qnew does not equal q is added to
        # the RRT; Trapped, in which the proposed new vertex
        # is rejected because it does not lie in Cfree

        nearest_node, distToNearestNode = T.find_nearest(q)
        # Direction vector from nearest node to the sample/(new configuration)
        direction = q - nearest_node.state # nearest_node.state is getting the value at nearest node.  Not to be confused with status.
        
        # length = np.linalg.norm(direction)
        # if _DEBUG:
        #     print('distToNearestNode', distToNearestNode)
        #     print('length', length)

        # Move a fixed step size.  Limit step length
        if distToNearestNode > self.epsilon:
            unitVector = (direction / distToNearestNode)
            direction = unitVector * self.epsilon 
        # Coordinate position 
        q_new = nearest_node.state + direction

        # Check for collision
        if not self.in_collision(q_new):  
            new_node = TreeNode(q_new)

            # add_node(newNode, parent)
            T.add_node(new_node, nearest_node)

            # Check if the sampled node was reached within epsilon, then return status.
            if distToNearestNode < self.epsilon:
                return _REACHED, new_node
            else:
                return _ADVANCED, new_node
        else:
            return _TRAPPED, None

    def fake_in_collision(self, q):
        '''
        We never collide with this function!
        '''
        return False

def test_rrt_env(num_samples=500, step_length=2, env='./env0.txt', connect=False):
    '''
    create an instance of PolygonEnvironment from a description file and plan a path from start to goal on it using an RRT

    num_samples - number of samples to generate in RRT
    step_length - step size for growing in rrt (epsilon)
    env - path to the environment file to read
    connect - If True run rrt_connect

    returns plan, planner - plan is the set of configurations from start to goal, planner is the rrt used for building the plan
    '''
    pe = PolygonEnvironment()
    pe.read_env(env)

    dims = len(pe.start)
    start_time = time.time()
    
    rrt = RRT(num_samples,
              dims,
              step_length,
              lims = pe.lims,
              connect_prob = 0.05,
              collision_func=pe.test_collisions)
    if connect:
        plan = rrt.build_rrt_connect(pe.start, pe.goal)
    else:
        plan = rrt.build_rrt(pe.start, pe.goal)
    run_time = time.time() - start_time
    print('plan:', plan)
    print( 'run_time =', run_time)

    pe.draw_env(show=False)
    pe.draw_plan(plan, rrt, True, True, True)

    return plan, rrt
