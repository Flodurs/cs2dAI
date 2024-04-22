import numpy as np

class tile:
    def __init__(self):
        self.typ = 0 #0 = boden; 1=mauer; 2=niedrige mauer;
        
        self.size = 40
        self.pos = np.zeros(2)
        
    def getTyp(self):
        return self.typ
        
    def setTyp(self,typ):
        self.typ = typ
        
    def setPos(self,pos):
        self.pos = pos
        
    def getRect(self):
        return [self.pos[0], self.pos[1], self.size, self.size]