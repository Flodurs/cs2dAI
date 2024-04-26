import torch
from torch import nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self,inputNum):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputNum+1, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
        
class qModel:
    def __init__(self,inputNum):
      
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        self.model = NeuralNetwork(inputNum).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.epochNum = 10
        
    def saveModel(self,path):
        pass
        
    def updateModel(self,states,actions,reward):
    
        
        for i in range(self.epochNum):
            inputs = []
            targetOutputs = []
            
            
            
            for i in range(len(states)):
                inputs.append(np.concatenate(([states[i],actions[i],1])))
                targetOutputs.append([reward])
                
            inputs = torch.FloatTensor(inputs) 
            targetOutputs = torch.FloatTensor(targetOutputs) 
             
            self.optimizer.zero_grad()
                
            outputs = self.model(inputs.to(self.device)).cpu()
            loss = self.criterion(outputs, targetOutputs)
            loss.backward()
            self.optimizer.step()
        
        
    def forward(self,input):
        return self.model(torch.FloatTensor(np.concatenate((input,[1]))).to(self.device))[0].cpu().detach().numpy()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    