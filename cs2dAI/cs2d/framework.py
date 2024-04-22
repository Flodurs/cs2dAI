import pygame
import cs2d.zeichner
import cs2d.baseWorld
import cs2d.baseAgent



class framework:
    def __init__(self):
        self.bRunning = True 
        self.renderer = cs2d.zeichner.zeichner()
               
      
        self.worldViewd = 0
        print("Init Framework")
        pygame.init()
        
    def run(self):
        while self.bRunning == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bRunning = False
        
        
            self.update()
            self.render(self.worldViewd)
            
    def update(self):
        pass
            
            
        
    def render(self,w):
        pass
        
f = framework()
f.run()