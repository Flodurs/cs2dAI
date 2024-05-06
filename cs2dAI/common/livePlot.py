import matplotlib.pyplot as plt

class livePlot:
    def __init__(self):
        self.data = []
        
    def draw(self):
        plt.plot(data)
        
    def drawAvgLast(self):
        avgData = []
        plt.clf()
        
        if len(self.data) < 40:
            return
        
        for i in range(0,int(len(self.data)-40)):
            avgData.append(sum(self.data[i:i+40]))
            
       
            
        plt.plot(avgData)
    
    def addData(self,d):
        self.data.append(d)
        
    def update(self):
        plt.pause(0.0000001)
        