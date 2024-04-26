import cs2d.baseAgent
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import reinforcement.qModel

import random
import time


class reinforcementLearningAgent(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
   
        self.actionNum = 6
        self.inputNum = 6 +self.actionNum
        
        self.epsilon = 0.9
        
        self.q = reinforcement.qModel.qModel(self.inputNum)
  
        
        self.actionsperformed = []
        self.states = []
        
    def think(self,world):
        state = self.getViewList().flatten()
        
        #choose Action
        action = 0
        actionInputs = [[0.0 if j != k else 1.0 for j in range(self.actionNum)] for k in range(self.actionNum)]
        if random.uniform(0,1) > self.epsilon:
            action = random.randrange(0,self.actionNum)
            
        else:
            
            moveEval = []
            inputs = np.zeros((self.actionNum,self.inputNum))
            for i,inp in enumerate(actionInputs):
                inputs[i] = np.concatenate((state,np.array(inp)))
            
            for i in range(self.actionNum):
                moveEval.append(self.q.forward(inputs[i]))
                
          
            action = np.argmax(moveEval)
   
        
        self.states.append(state)
        self.actionsperformed.append(actionInputs[action])
        #execute Actions
        self.executeAction(action)
        
        
        
        
        #update Q-Net on reward
        if self.pos[1] > 400:
            self.q.updateModel(self.states,self.actionsperformed,1/len(actionInputs))
            print("update")
            self.pos = [200,100]
        
        
        
        
        
        
        