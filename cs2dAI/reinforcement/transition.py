class transition:
    def __init__(self):
        self.state = []
        self.nextStage = []
        self.action = 0
        self.reward = 0
        
    def getState(self):
        return self.state
        
    def getNextState(self):
        return self.nextStage
    
    def getAction(self):
        return self.action
    
    def getReward(self):
        return self.reward
        
   