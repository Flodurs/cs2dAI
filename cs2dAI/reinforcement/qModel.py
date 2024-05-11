import torch
from torch import nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class NeuralNetwork(nn.Module):
    def __init__(self,inputNum,outputNum):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputNum+1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, outputNum),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
        
class qModel:
    def __init__(self,inputNum,outputNum):
    
        self.REPLAY_MEMORY_SIZE = 50000
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINI_BATCH_SIZE = 32
        self.DISCOUNT = 0.99
        self.UPDATE_TARGET_INTERVAL = 5
      
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        self.model = NeuralNetwork(inputNum,outputNum).to(self.device)
        self.targetModel = NeuralNetwork(inputNum,outputNum).to(self.device)
        self.targetModel.load_state_dict(self.model.state_dict())
        
        self.targetUpdateCounter = 0
        
        self.replayMemory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        
    
        
    def saveModel(self,path):
        pass
        
    def updateModel(self,states,actions,reward):
        for i in range(self.epochNum):
            inputs = []
            targetOutputs = []
            
            
            for i in range(len(states)):
                inputs.append(np.concatenate(([states[i],actions[i],[1]])))
                
                
            inputs = torch.FloatTensor(inputs) 
            outputs = self.model(inputs.to(self.device)).cpu()  
            
            for i in range(len(states)):
                targetOutputs.append([outputs[i][0]+1/len(actions)*(reward-outputs[i][0])])
            
            
            #print(targetOutputs)
            
            targetOutputs = torch.FloatTensor(targetOutputs) 
             
            self.optimizer.zero_grad()
                
            
            loss = self.criterion(outputs, targetOutputs)
            loss.backward()
            self.optimizer.step()
            
    def  updateFromMultipleEpisodes(self,episodes):
    
        inputs = []
        targetOutputs = []
        rewards = []
    
        for ep in episodes:
            for i in range(ep.getLen()):
                inputs.append(np.concatenate((ep.getStates()[i],ep.getActions()[i],[1])))
                rewards.append(ep.getReward())
            
        inputs = torch.FloatTensor(inputs)
        #print(len(inputs[0]))
        outputs = self.model(inputs.to(self.device)).cpu()  
        
        for ep in episodes:
            for i in range(ep.getLen()):
                targetOutputs.append([outputs[i][0]+1/ep.getLen()*(rewards[i]-outputs[i][0])])
            
            
        targetOutputs = torch.FloatTensor(targetOutputs)

        
    def forward(self,input):
        return self.model(torch.FloatTensor(np.concatenate((input,[1]))).to(self.device)).cpu().detach().numpy()
    
    def forwardTargetModel(self,input):
        return self.targetModel(torch.FloatTensor(np.concatenate((input,[1]))).to(self.device)).cpu().detach().numpy()
    
    def updateReplayMemory(self, transition):
        self.replayMemory.append(transition)
        
    def train(self, terminalState):
        if len(self.replayMemory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        
        batch = random.sample(self.replayMemory, self.MINI_BATCH_SIZE)
        
        currentStates = [transition[0] for transition in batch]
        nextStates = [transition[3] for transition in batch]
        
        currentQsList = []
        nextQsList = []
        
        for i in range(len(currentStates)):
            currentQsList.append(self.forward(currentStates[i]))
            nextQsList.append(self.forwardTargetModel(nextStates[i]))
        
        x = []
        y = []
        
        for i,(currentState, action, reward, nextState, done) in enumerate(batch):
            if not done:
                maxNextQ = np.max(nextQsList[i])
                newQ = reward + self.DISCOUNT * maxNextQ
            else:
                newQ = reward
    
            
            currentQs = currentQsList[i]
            currentQs[action] = newQ
            
            y.append(currentQs)
            
       
        self.optimizer.zero_grad()
        loss = self.criterion(torch.FloatTensor(currentQsList[i]), torch.FloatTensor(y[i]))
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
        
        
      
        

        if self.targetUpdateCounter > self.UPDATE_TARGET_INTERVAL:
            self.targetModel.load_state_dict(self.model.state_dict())
            self.targetUpdateCounter = 0
            print("up")
    
            
    def incTargetUpdateCounter(self):
        self.targetUpdateCounter+=1
        
    
    
    
    
    
    
    