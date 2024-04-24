import cs2d.baseAgent
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np

import random
import time


class NeuralNetwork(nn.Module):
    def __init__(self,inputNum):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputNum, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits




class reinforcementLearningAgent(cs2d.baseAgent.baseAgent):
    def __init__(self):
        super().__init__()
        
        self.actionNum = 4
        self.inputNum = 6 +self.actionNum
        
        self.epsilon = 0.5
        
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        self.model = NeuralNetwork(self.inputNum).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        actionsperformed = []
        
    def think(self,world):
        state = self.getViewList().flatten()
        
        print(state)
        
        print("----------")
        
        
        
        #choose Action
        
        
        if random.uniform(0,1) > self.epsilon:
            action = random.randrange(0,self.actionNum)
        else:
            actionInputs = [[0.0 if j != k else 1.0 for j in range(self.actionNum)] for k in range(self.actionNum)]
            moveEval = []
            inputs = np.zeros((self.actionNum,self.inputNum))
            for i,inp in enumerate(actionInputs):
                print(state)
                print(np.array(inp))
                inputs[i] = np.concatenate((state,np.array(inp)))
            print(actionInputs)
            for i in range(self.actionNum):
                moveEval.append(self.model(torch.FloatTensor(inputs[i]).to(self.device))[0].cpu().detach().numpy())
                
            print("aa")
            action = np.argmax(moveEval)
   
        
        #execute Action
        self.executeAction(action)
        print("aa")
        
        
        
        #update Q-Net on reward
        
        
        
        
        
        
        
        
        