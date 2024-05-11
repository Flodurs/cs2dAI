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


import common.livePlot


class reinforcementLearningAgent(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
   
        self.actionNum = 6
        self.inputNum = 18
        
        self.lP = common.livePlot.livePlot()
        
        self.START_EPSILON = 0.1
        self.END_EPSILON = 0.0001
        self.DECAY_EPSILON = 0.0001
        
        self.epsilon = self.START_EPSILON
        
        self.q = reinforcement.qModel.qModel(self.inputNum,self.actionNum)
  
        
        self.episodes = []
        self.episodes.append(reinforcement.episode.episode())
        
        self.step = 0
        self.epNum = 0
        self.globalStep = 0
        self.rewardHistory = []
        
        self.lastState = [0 for i in range(9)]
        
        
        
    def think(self,world):
        self.lP.update()
        
        state = self.getViewList()
        state = self.processInputs(state)
        
        
        totalState = self.lastState + state
        #print(totalState)
        
        reward = 0
        done = False
        
        #choose Action
        action = 0
        self.step+=1
        self.globalStep+=1
       
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(0,self.actionNum)
            #print("Random")
        else:
            action = np.argmax(self.q.forward(totalState))
                
        #execute Actions
        for s in range(len(state)):
            self.lastState[s] = state[s]
        
        
        self.executeAction(action)
        
        newState = self.getViewList()
        newState = self.processInputs(newState)
        
        newState+=self.lastState
        
        
        if self.pos[1] > 300:
            reward = 1
            self.pos = [200,100]
            self.actionsperformed = []
            self.states = []
            self.rotation = 0
            self.resetPhysics()
            done = True
            self.step = 0
            print("Epsilon: " +str(self.epsilon))
            
            self.epNum+=1
            self.rewardHistory.append(1)
            self.lP.addData(reward)
            self.lP.drawAvgLast()
            
            if self.epsilon > self.END_EPSILON:
                self.epsilon -= self.DECAY_EPSILON
            
        if self.step > 500:
            self.step = 0
            reward = 0
            self.pos = [200,100]
            self.actionsperformed = []
            self.states = []
            self.rotation = 0
            self.resetPhysics()
            done = True
            print("Epsilon: " + str(self.epsilon))
            self.epNum+=1
            self.rewardHistory.append(0)
            self.lP.addData(reward)
            self.lP.drawAvgLast()
           
            if self.epsilon > self.END_EPSILON:
                self.epsilon -= self.DECAY_EPSILON
            
        if self.globalStep%100 == 0:
            self.q.incTargetUpdateCounter()
        
        self.q.updateReplayMemory([totalState,action,reward,newState,done])
        if self.globalStep%5 == 0:
            self.q.train(done)
        
        
        
       
      
        