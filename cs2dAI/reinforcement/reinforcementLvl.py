import cs2d.baseWorld
import reinforcement.reinforcementLearningAgent
import numpy as np

class reinforcementLvl(cs2d.baseWorld.baseWorld):
    def __init__(self):
        super().__init__()
        self.agents.append(reinforcement.reinforcementLearningAgent.reinforcementLearningAgent(np.array([200.0,200.0])))
        self.agents.append(reinforcement.reinforcementLearningAgent.reinforcementLearningAgent(np.array([200.0,200.0])))
        