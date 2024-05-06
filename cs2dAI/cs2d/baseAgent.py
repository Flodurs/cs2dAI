import math 
import cs2d.geometry
import numpy as np

import random

class baseAgent:
    def __init__(self,pos):
        random.seed()
        #physics
        self.size = 40
        self.pos = pos
        self.vel = np.array([0.0,0.0])
        self.angularVel = 0
        self.rotation = 0 #[0,2Pi[ 0 facing +y; Pi/2 facing right
        
        self.initPos = np.array([1020,600])
        
        #view
        self.viewRayNum = 3
        self.viewRayAngles = [-0.3,0,0.3] 
        self.viewList = np.zeros((self.viewRayNum,2))
        self.viewRange = 800
        
        #thinking
     
        self.marked = 0
        
        self.range = 800
        self.shotFlag = 0
        self.shotRange = -1
        self.shotPos = np.array([0,0])
        self.shotDir = np.array([0,0])
        self.weaponCoolDown = 10
        self.weaponTimer = 0
        self.hp = 100
        
        
    def setMarked(self):
        self.marked = 1
    
    def getMarked(self):
        return self.marked
        
    def getShotFlag(self):
        return self.shotFlag
        
        
    def update(self,world,deltaTime):
        self.updateViewList(world)
        self.updateWeapon(deltaTime)
        self.shotFlag = 0
        
        self.think(world)
       
       

        
        self.pos += self.vel*deltaTime*200
        if self.pos[0] < 10:
            self.pos[0] = 10
        if self.pos[0] > 790:
            self.pos[0] = 790
        if self.pos[1] < 10:
            self.pos[1] = 10
        if self.pos[1] > 790:
            self.pos[1] = 790
        
        
        #self.vel=np.clip(self.vel,-2,2)
        
        self.rotation+=self.angularVel*deltaTime*10
        if self.handleCollision(world) == True:
            self.pos -= self.vel*deltaTime*200
            

    def setConnectionMatrix(self,matrix):
        self.brain.setConnectionMatrix(matrix)
        
        
    def setPos(self,pos):
        self.pos = pos
    
    def getViewRange(self):
        return self.viewRange
        
    def getShotPos(self):
        return self.shotPos
        
    def getShotDir(self):
        return self.shotDir
        
    def getPos(self):
        return self.pos
        
    def getSize(self):
        return self.size
        
    def getSpeed(self):
        return np.linalg.norm(self.vel)
        
    def getFacingVector(self):
        return np.array([math.sin(self.rotation),math.cos(self.rotation)])
        
    # slow    
    # def rayCast(self,direction,length,world):
        # for wall in world.getWalls():
            # sps = geometry.lineRectIntersect([self.pos[0],self.pos[1],self.pos[0]+direction[0]*length,self.pos[1]+direction[1]*length],wall.getRect())
            # dists = []
            # #print(sps)
            # for sp in sps:
                # dists.append(np.linalg.norm(sp-self.pos))
            # if len(dists) > 0:
                # return min(dists)
        # return -1
        
    def rayCast(self,direction,length,world):
        samplingDist = 20
        sampleNum = length/samplingDist
        for i in range(int(sampleNum)):
            px=self.pos[0]+i*direction[0]*samplingDist
            py=self.pos[1]+i*direction[1]*samplingDist
            for wall in world.getWalls():
                if cs2d.geometry.pointInRect(px,py,wall):
                    return np.linalg.norm(np.array([px,py])-self.pos)
        return -1
        
    def rayCastWall(self,direction,length,world):
        samplingDist = 30
        sampleNum = length/samplingDist
        for i in range(int(sampleNum)):
            px=self.pos[0]+i*direction[0]*samplingDist
            py=self.pos[1]+i*direction[1]*samplingDist
            for wall in world.getWalls():
                if cs2d.geometry.pointInRect(px,py,wall):
                    return(np.linalg.norm(np.array([px,py])-self.pos))
    
        return float('inf')
        
    def rayCastAgent(self,direction,length,world):
        samplingDist = 30
        sampleNum = length/samplingDist
        for i in range(2,int(sampleNum)):
            px=self.pos[0]+i*direction[0]*samplingDist
            py=self.pos[1]+i*direction[1]*samplingDist
            for ag in world.getAgents():
                #print(ag.getRect())
                if cs2d.geometry.pointInRect(px,py,ag.getRect()):
                    
                    agentDist=np.linalg.norm(np.array([px,py])-self.pos)
                    
                    return [agentDist,ag]
        return float('inf'),-1
        
        
    def rayCastWallandAgents(self,direction,length,world):
        
        
        wallDist = self.rayCastWall(direction,length,world)
        
        
        agentDist,agent = self.rayCastAgent(direction,length,world)
        
        
                    
        if wallDist == float('inf') and agentDist == float('inf'):
            return [-1,-1]

        if wallDist < agentDist:
            return [0,wallDist]
        else:
            return [1,agentDist,agent]
                    
    def getViewList(self):
        return self.viewList
        
    
        
        
    def updateViewList(self,world):
        dists = []
        types = [] #-1 nothing; 0 wall; 1 agent
        
        
        
        for i,ray in enumerate(self.viewRayAngles):
            rayCastResult = self.rayCastWallandAgents(cs2d.geometry.angleToVec(self.rotation+ray), self.viewRange, world)
            
        
            self.viewList[i][0] = rayCastResult[1]
            self.viewList[i][1] = rayCastResult[0]
            
    def getRect(self):
        return [self.pos[0]-self.size/2,self.pos[1]-self.size/2,self.size,self.size]

    def handleCollision(self,world):
        for wall in world.getWalls():
            if cs2d.geometry.circleRectIntersect([self.pos[0], self.pos[1],40], wall):
                #self.vel = np.array([0,0])
                return True
        return False
                
    def getViewRayDirections(self):
        directions = []
        for angle in self.viewRayAngles:
            directions.append(cs2d.geometry.angleToVec(self.rotation+angle))
        return directions
        
    def resetPhysics(self):
        self.vel = np.array([0.0,0.0])
        self.angularVel = 0
        
    #--------------------------------------------------AI-----------------------------------------------------
    #netStructure 5 input neurons: distances mapped to [0,1]
    #6 output neurons: up,down,left,right, turn+, turn-
    
    
    
    #overwwrite with agent Code
    # def think(self,world):
        # pass
        
           
         
            
        
    def processInputs(self,inputs):
        processedInputs = []
        for inp in inputs:
            processedInputs.append(inp[0]/self.viewRange+1)
            processedInputs.append(inp[1])
            
            
        if self.rotation >= 0:
            processedInputs.append(math.fmod(self.rotation,2*math.pi))
        if self.rotation < 0:
            processedInputs.append(-math.fmod(self.rotation,2*math.pi))
            
        processedInputs.append(self.pos[0]/800)
        processedInputs.append(self.pos[1]/800)
    
        
        
        
        #print(len(processedInputs))
        return processedInputs
        
    def actFromOutputs(self,outputs,world):
        acc = np.array([0.0,0.0])
        if outputs[0] > 0.5:
            #print("up")
            acc += np.array([0.0,-1.0])
            
        if outputs[1] > 0.5:
            #print("down")
            acc += np.array([0.0,1.0])
            
        if outputs[2] > 0.5:
            #print("left")
            acc += np.array([-1.0,0.0])
            
        if outputs[3] > 0.5:
            #print("right")
            acc += np.array([1.0,0.0])
            
        #print(acc)
            # print(self.rotation)
            
        self.vel+=np.matmul(np.array([[math.cos(self.rotation),math.sin(self.rotation)],[-math.sin(self.rotation),math.cos(self.rotation)]]),acc)
        self.vel = self.normalize(self.vel)
        
        rotAcc = 0
            
        if outputs[4] > 0.5:
            rotAcc += 0.1
            
            #print("left")
            
        if outputs[5] > 0.5:
            rotAcc -= 0.1
            
        self.angularVel+=rotAcc
        self.angularVel=np.clip(self.angularVel,-0.5,0.5)
        
            #print("right")
            
        if outputs[6] > 0.5:
            self.shoot(world)
            
            
    def executeAction(self,a):
        acc = np.array([0.0,0.0])
        angAcc = 0
      
        if a == 0:
            #print("up")
            acc += np.array([0.0,-1.0])
            
        if a == 1:
            #print("down")
            acc += np.array([0.0,1.0])
            
        if a == 2:
            #print("left")
            acc += np.array([-1.0,0.0])
            
        if a == 3:
            #print("right")
            acc += np.array([1.0,0.0])
            
        if a==4:
            angAcc += 0.1
            
        if a==5:
            angAcc -= 0.1
            
        self.vel+=np.matmul(np.array([[math.cos(self.rotation),math.sin(self.rotation)],[-math.sin(self.rotation),math.cos(self.rotation)]]),acc)
        self.vel = self.normalize(self.vel)
        
        self.angularVel+=angAcc
        self.angularVel = np.clip(self.angularVel,-1,1)
            
    #---------------------------combat mechanics-------------------------------------------
    
    def getShotRange(self):
        return self.shotRange
            
    def hit(self):
        self.hp = 0
    
    def getHp(self):
        return self.hp
            
    def shoot(self,world):
        if self.weaponTimer > 0.5:
            self.shotFlag = 1
            self.weaponTimer = 0
            self.shotPos = self.pos
            self.shotDir = self.getFacingVector()
            rayCastResult = self.rayCastWallandAgents(geometry.angleToVec(self.rotation),self.range, world)
            
            if rayCastResult[0] == -1:
                self.shotRange = self.range
            else:
                self.shotRange = rayCastResult[1] 
            
            if rayCastResult[0]==1:
                #print("hit")
                rayCastResult[2].hit()
        
    def updateWeapon(self,deltaT):
        self.weaponTimer += deltaT 
        
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
        
        
     