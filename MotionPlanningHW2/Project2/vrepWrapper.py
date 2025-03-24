# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:34:14 2019

@author: tabor
"""

import sys
import time
import numpy as np
import os

try:
    from vrepfiles.vrepfiles import vrep
except:
    print(
        "Import of vrep failed. Make sure the 'vrep.py' file is in this directory."
    )
    sys.exit(1)
    

class vrepWrapper:
    def __init__(self,shm = False,sa=None):       
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        self.start = np.array([0.0] * 7)
        self.goal = np.array([3.14/2] * 7)
        pi = np.pi
        self.lims = np.array([[-pi, pi], [-pi, pi], [-pi, pi],
                              [-pi, pi], [-pi, pi], [-pi, pi],
                              [-pi, pi]])
        

        if self.clientID != -1:
            print("Successfully connected to remote API server.")
            #vrep.simxSynchronous(self.clientID, True)
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        else:
            print("Failed connecting to remote API server")
                                                       
        joint_names = ["LBR4p_joint{}".format(i) for i in range(1, 8)]
        self. joint_handles = [ vrep.simxGetObjectHandle(self.clientID, joint, vrep.simx_opmode_blocking)[1]
        for joint in joint_names]
     
        self.sa=sa

        self.setCollision(shm)
        self.shm = shm


    def setCollision(self,shm):
        if(shm == True):
            self.array = self.sa.attach("shm://test5")
            self.test_collisions = self.fastCollision
        else:
            self.test_collisions = self.slowCollisions

    def runTrajectory(self,angles):
        for angle in angles:
            for i in range(7):
                vrep.simxSetJointPosition(self.clientID, self.joint_handles[i], angle[i],
                                                vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.clientID)
            time.sleep(0.01)
    def checkCollission(self,states):
        num_states = len(states)
        single_dim_states = np.reshape(states,-1)

        [res, retInts, retFloats, retStrings, retBuffer] = vrep.simxCallScriptFunction(self.clientID, "LBR4p",
                                                                 vrep.sim_scripttype_childscript, 'collision_run',[num_states],single_dim_states,
                                                                 'Hello world!', 'blah', vrep.simx_opmode_blocking)
        if( res&2 == 2):
            print('deeper')
            new = np.array_split(states,2)
            collides1,fk1 = self.checkCollission(new[0])
            collides2,fk2 = self.checkCollission(new[1])
            retFloats = np.concatenate([fk1,fk2],0)
            retInts = collides1.extend(collides2)

            

        fk = np.reshape(np.array(retFloats),(-1,3))
        return (retInts,fk)
    #assumes (num,6) array (x1,y1,z1,x2,y2,z2)
    def addLine(self,lines,isPlan = 0):
        print(lines.shape)
        num_lines = len(lines)
        single_dim_lines = np.reshape(lines,-1)
        [res, retInts, retFloats, retStrings, retBuffer] = vrep.simxCallScriptFunction(self.clientID, "LBR4p",
                                                         vrep.sim_scripttype_childscript, 'addLines',[num_lines,isPlan],single_dim_lines,
                                                         'Hello world!', 'blah', vrep.simx_opmode_blocking)
    def addPoint(self,points,isEnd = 0):
        num_points  = len(points)
        single_dim_points = np.reshape(points,-1)
        [res, retInts, retFloats, retStrings, retBuffer] = vrep.simxCallScriptFunction(self.clientID, "LBR4p",
                                                         vrep.sim_scripttype_childscript, 'addPoints',[num_points,isEnd],single_dim_points,
                                                        'Hello world!', 'blah', vrep.simx_opmode_blocking)
    
    def vrepStop(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)

    def vrepReset(self):
        self.vrepStop()
        self.__init__(self.shm,self.sa)
        self.vrepStop()
        self.__init__(self.shm,self.sa)
        
    def slowCollisions(self,state):

        formatted = np.reshape(np.array(state),(-1,7))
        collides,fk = self.checkCollission(formatted)
        return np.sum(collides)>0

    def fastCollision(self,state):

        self.array[1:8] = state
        self.array[0] = 1
        while(self.array[0] == 1):
            #print(self.array[0:8])
            #print(self.array[8:])
            pass
        return(self.array[8]==1.0)        
        
    def draw_plan(self, plan, planner, dynamic_tree=False, dynamic_plan=True, show=True):
        if(self.shm):
           self.vrepReset()
        #dynamic_tree, dynamic_plan and show are all dummy values and do not function
        planLines = np.zeros((len(plan),6))

        collisions,fk = self.checkCollission(np.array(plan))        
        if(np.sum(collisions) > 0):
            print('plan failed checker')
        for i in range(len(plan)):
            if(i >0):
                planLines[i-1,3:] = fk[i,:]
            planLines[i,:3] = fk[i,:]
        self.addLine(planLines[:-1],1)
        
        if planner is not None:
            print('drawing tree')
            Qs, edges = planner.T.get_states_and_edges()
            totalNodes = np.reshape(np.array(edges),(-1,7))
            collisions,fk = self.checkCollission(totalNodes)

            if(np.sum(collisions) > 0):
                print('some nodes are in collision in tree')
            self.addLine(np.reshape(fk,(-1,6)))
            self.addPoint(fk)
#        return fk
        col,fk = self.checkCollission(np.reshape(planner.goal,(-1,7)))
        self.addPoint(np.reshape(fk,(-1,3)),1)          
        self.runTrajectory(plan)
        

