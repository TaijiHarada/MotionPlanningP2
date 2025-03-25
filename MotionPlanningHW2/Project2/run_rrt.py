# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:17:29 2019

@author: tabor
"""
import numpy as np
import matplotlib.pyplot as plotter
from math import pi
from collisions import PolygonEnvironment
import time
import vrepWrapper
from rrt import *

DEBUG = True
connect = False
bidirection = False
num_samples= 500
# problem = "vrep"
problem = './project2/env0.txt'

np.random.seed(0)

#load problem
if(problem == "vrep"):
    environment = vrepWrapper.vrepWrapper()
    step_length=0.1
else:
    environment = PolygonEnvironment()
    environment.read_env(problem)
    step_length=2


# Get the number of dimensions from the start position.
dims = len(environment.start)
start_time = time.time()

rrt = RRT(num_samples,
          dims,
          step_length,
          lims = environment.lims,
          connect_prob = 0.05,
          collision_func=environment.test_collisions)
if connect:
    plan = rrt.build_rrt_connect(environment.start, environment.goal)
elif bidirection:
    plan = rrt.build_bidirectional_rrt_connect(environment.start, environment.goal)
else:
    if DEBUG:
        print('start:', environment.start)
        print('goal:', environment.goal)

    plan = rrt.build_rrt(environment.start, environment.goal)

#if(problem == "vrep"):
#    environment.vrepReset()

run_time = time.time() - start_time
print('plan:', plan)
print('run_time =', run_time)

# Params draw_plan(self, plan, planner, dynamic_tree=False, dynamic_plan=True, show=True): IDK what 3rd True paramater does.
debugThing = environment.draw_plan(plan, rrt,True,True,True)


if(problem == "vrep"):
    time.sleep(10)
    environment.vrepStop()

