class episode:
    def __init__(self):
        self.reward = 0
        self.states = []
        self.actions = []
        
    def addStep(self, state,action):
        self.states.append(state)
        self.actions.append(action)
    
    def getStates(self):
        return self.states
        
    def getActions(self):
        return self.actions
        
    def getReward(self):
        return self.reward
        
    def getLen(self):
        return len(self.states)
        
    def setReward(self,r):
        self.reward = r
        