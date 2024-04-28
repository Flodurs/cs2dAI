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


class reinforcementLearningAgent(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
   
        self.actionNum = 6
        self.inputNum = 8 
        
        self.epsilon = 0.2
        
        self.q = reinforcement.qModel.qModel(self.inputNum,self.actionNum)
  
        
        self.episodes = []
        self.episodes.append(reinforcement.episode.episode())
        
    def think(self,world):
        state = self.getViewList()
        state = self.processInputs(state)
        reward = 0
        done = False
        
        #choose Action
        action = 0
       
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(0,self.actionNum)
            #print("Random")
        else:
            action = np.argmax(self.q.forward(state))
                
        #execute Actions
        self.executeAction(action)
        
        newState = self.getViewList()
        newState = self.processInputs(newState)
        
        if self.pos[1] > 400:
            reward = 1
            self.pos = [200,100]
            self.actionsperformed = []
            self.states = []
            self.rotation = 0
            self.resetPhysics()
            done = True
            
        if self.episodes[-1].getLen() > 1000:
            reward = 0
            self.pos = [200,100]
            self.actionsperformed = []
            self.states = []
            self.rotation = 0
            self.resetPhysics()
            done = True
        
        self.q.updateReplayMemory([state,action,reward,newState,done])
        self.q.train(done)
        
        
        
       
      
        