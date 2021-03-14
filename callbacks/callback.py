class Callback:
    def __init__(self) -> None:
        super().__init__()
        self.agent = None

    def _begin_fit(self, agent, **kwargs):
        self.agent = agent
    
    def _after_fit(self, **kwargs):
        pass

    def _begin_next(self, **kwargs):
        pass

    def _after_next(self, **kwargs):
        pass
    

