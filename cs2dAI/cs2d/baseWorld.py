#tile based world
import cs2d.tile
import cs2d.baseAgent
import numpy as np

import random
#import zeichner
import time

class baseWorld:
    def __init__(self):
        self.iSize = 20
        self.iTileSize = 40
        self.aTiles = [[cs2d.tile.tile() for j in range(self.iSize)] for k in range(self.iSize)]
        self.aWalls = []
      
        self.agents = []
        
        
   
        

        
        
        #--------------------------physics------------
        self.simulationTime = 0 #time used for sim
        self.deltaT = 0.02
        
    def recalcWalls(self):
        self.aWalls = []
        for j in range(self.iSize):
            for k in range(self.iSize):
                if self.aTiles[j][k].getTyp() == 1:
                    self.aTiles[j][k].setPos(np.array([j*self.iTileSize,k*self.iTileSize]))
                    self.aWalls.append(self.aTiles[j][k])
        
    def getWalls(self):
        return self.aWalls
        
    def update(self):
        return self.updateAgents()

    def updateAgents(self):
        if len(self.agents) <= 0:
            return -1
            
        for ind,ag in enumerate(self.agents):
            ag.update(self,self.deltaT)
           
            if ag.getHp() <= 0:
                return ind
        
        return -1
            
    def getTiles(self):
        return self.aTiles
        
    def getSize(self):
        return self.iSize
    
    def getAgents(self):
        return self.agents
        
    def addAgent(self,agent):
        self.agents.append(agent)
        
    #overwrite with playMatch function    
    def playMatch(self,matrixA,matrixB):
        pass
        
    
        

        
        
       
        
    