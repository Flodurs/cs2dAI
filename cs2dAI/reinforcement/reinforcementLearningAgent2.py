import cs2d.baseAgent
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import reinforcement.qModel
import reinforcement.episode

import random
import time

import itertools

import copy 

import common.livePlot

from collections import deque


class reinforcementLearningAgent2(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
   
       
        
        self.lP = common.livePlot.livePlot()
        
        self.START_EPSILON = 1
        self.END_EPSILON = 0.1
        self.DECAY_EPSILON = 0.99
        self.PAST_STATES_NUM = 5
        self.FRAMES_EXPOSED = 5
        
        self.inputNumPerFrame = 9
        self.actionNum = 6
        self.inputNum = self.inputNumPerFrame*self.FRAMES_EXPOSED
        
        
        
        
        self.epsilon = self.START_EPSILON
        
        self.q = reinforcement.qModel.qModel(self.inputNum,self.actionNum)
  
        
        self.step = 0
        self.global_step = 0
     
        
        self.lastAction = 0
        
        self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
        self.lastStateCollection = []
        
        for i in range(self.FRAMES_EXPOSED):
            self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            
        
    def think(self,world):
        self.lP.update()
        
        
        current_state = self.getViewList()
        current_state = self.processInputs(current_state)
        
        self.stateCollection.append(current_state)
        
     
        
        reward = 0
        done = False
        
        #check for reward
        if self.pos[1] > 300:
            reward = 1
            self.pos = [200,100]
          
          
            self.rotation = 0
            self.resetPhysics()
            done = True
            
            print("Epsilon: " +str(self.epsilon))
            
            self.lP.addData(reward)
            self.lP.drawAvgLast()
            
            if self.epsilon > self.END_EPSILON:
                self.epsilon *= self.DECAY_EPSILON
            
        if self.step > 500:
           
            reward = 0
            self.pos = [200,100]
            
            self.rotation = 0
            self.resetPhysics()
            done = True
            print("Epsilon: " + str(self.epsilon))
            self.step = 0
          
            self.lP.addData(reward)
            self.lP.drawAvgLast()
           
            if self.epsilon > self.END_EPSILON:
                self.epsilon *= self.DECAY_EPSILON
        
        
       
        #update Replay Memory 
        if self.global_step != 0:
            self.q.updateReplayMemory([list(itertools.chain.from_iterable(self.lastStateCollection)),self.lastAction,reward,list(itertools.chain.from_iterable(self.stateCollection)),done])
        
        #choose Action
        action = 0
       
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(0,self.actionNum)
            #print("Random")
        else:
            action = np.argmax(self.q.forward(list(itertools.chain.from_iterable(self.stateCollection))))
                

        #store for transition
        self.lastAction = action
        self.lastState = []
        self.lastStateCollection = copy.deepcopy(self.stateCollection)
       
        self.executeAction(action)
            
        if self.global_step%400 == 0:
            self.q.incTargetUpdateCounter()
        
        if self.global_step%2 == 0:
            self.q.train(done)
            
        self.step+=1
        self.global_step+=1
        

        
       
      
        