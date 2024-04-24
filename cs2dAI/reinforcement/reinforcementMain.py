import reinforcement.reinforcementLvl
import reinforcement.reinforcementLearningAgent
import playground.playgroundLvl


class playgroundMain:
    def __init__(self):
        self.bRunning = True 
        self.w = playground.playgroundLvl.playgroundLvl()

    def run(self):
        while self.bRunning == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bRunning = False
            self.update()
            self.render()
            
    def update(self):
        self.w.update()
            
    def render(self):
        self.renderer.render(self.w)
        
    