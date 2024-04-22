import pygame
import cs2d.zeichner
import cs2d.baseWorld
import cs2d.baseAgent
import playground.playgroundLvl


class playgroundMain:
    def __init__(self):
        self.bRunning = True 
        self.renderer = cs2d.zeichner.zeichner()
               
        self.w = cs2d.baseWorld.baseWorld()
        self.w = playground.playgroundLvl.playgroundLvl()
        
        
        self.worldViewd = 0
      
        pygame.init()
        
    def run(self):
        
        while self.bRunning == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bRunning = False
        
        
            self.update()
            self.render(self.worldViewd)
            
    def update(self):
        self.w.update()
            
    def render(self,w):
        self.renderer.render(self.w)