import pygame
import numpy as np

class shootingAnimation():
    def __init__(self,pos,dir,shotRange):
        self.x=pos[0]
        self.y=pos[1]
        self.dir = dir
        self.timeLeft = 2
        self.shotRange = shotRange
        
        
        
        
    def update(self,screen):
        self.timeLeft-=0.1
        #print(self.timeLeft)
        if self.timeLeft >= 0:
            pygame.draw.line(screen, (0,0,self.timeLeft*100), [self.x,self.y], np.array([self.x,self.y])+self.dir*self.shotRange, 4)
            
    def getTimeLeft(self):
        return self.timeLeft
        


class zeichner:
    def __init__(self):
    
        self.screenSize = (1000, 900)
        
        
        self.screen = pygame.display.set_mode(self.screenSize)
        self.tileColors = [(100,100,100),(255,140,0)] #floor; wall
        self.tileSize = 40
        
        self.shootAnimations = []
        self.xOffset = 50
        self.yOffset = 50
        
        self.canvas = pygame.Surface(self.screenSize)
        

    def render(self,world):
        agents = world.getAgents()
        self.canvas.fill((0, 200, 0))
        self.screen.fill((0, 200, 0))
        self.drawWorld(world)
        self.drawAgents(agents)
        self.drawWalls(world)
        
        self.screen.blit(self.canvas,(self.xOffset,self.yOffset))
        
        pygame.display.flip()
        
    def drawWorld(self,world):
        tiles = world.getTiles()
        for j in range(world.getSize()):
            for k in range(world.getSize()):
                pygame.draw.rect(self.canvas, self.tileColors[tiles[j][k].getTyp()], pygame.Rect(j*self.tileSize, k*self.tileSize, self.tileSize, self.tileSize))
        
        #horizontal lines
        for i in range(world.getSize()+1):
            pygame.draw.line(self.canvas,(60,60,60),(0,i*self.tileSize),(world.getSize()*self.tileSize,i*self.tileSize),3)
        for i in range(world.getSize()+1):
            pygame.draw.line(self.canvas,(60,60,60),(i*self.tileSize,0),(i*self.tileSize,world.getSize()*self.tileSize),3)
            
            
    def drawWalls(self,world):
        for wall in world.getWalls():
            pygame.draw.rect(self.canvas,self.tileColors[1],wall)
            
            
    def drawAgents(self,agents):
    
        for ag in agents:
            if ag.getShotFlag():
                self.shootAnimations.append(shootingAnimation(ag.getShotPos(),ag.getShotDir(),ag.getShotRange()))
                
        for shot in self.shootAnimations:
            shot.update(self.screen)
            if shot.getTimeLeft() <= 0:
                self.shootAnimations.remove(shot)
    
    
    
    
        for ag in agents:
            
           
            pygame.draw.line(self.screen, (0,0,0), ag.getPos(), ag.getPos()+ag.getFacingVector()*25, 2)
            
            ranges = ag.getViewList()
            #print(ranges)
            for i in range(len(ag.getViewRayDirections())):
                if ranges[i][1] == -1:
                    color = (255,255,255)
                    if i != -99:
                        pygame.draw.line(self.screen,color , ag.getPos(), ag.getPos()+ag.getViewRayDirections()[i]*ag.getViewRange(), 1)
                elif ranges[i][1] == 0:
                    color = (255,0,0)
                    if i != -99:
                        pygame.draw.line(self.screen,color , ag.getPos(), ag.getPos()+ag.getViewRayDirections()[i]*ranges[i][0], 1)
                else:
                    color = (0,255,0)
                    if i != -99:
                        pygame.draw.line(self.screen,color , ag.getPos(), ag.getPos()+ag.getViewRayDirections()[i]*ranges[i][0], 1)
                
            
            
            pygame.draw.circle(self.screen, (255, 0, 0), ag.getPos(), ag.getSize()/2)
            
        for ag in agents:
            if ag.getSpeed() <= 0.4:
                pygame.draw.circle(self.screen, (0, 255, 0), ag.getPos(), ag.getSize()/2)
    
    