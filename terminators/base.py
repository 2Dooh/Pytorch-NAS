from abc import abstractmethod, ABC

import time

class Terminator(ABC):
    def __init__(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def _criteria_met(self, **kwargs):
        raise NotImplementedError

class ConditionMet(Terminator):
    def __init__(self, counter, flag, **kwargs):
        super().__init__(counter=counter, flag=flag, **kwargs)
        self.flag = flag
        self.counter = counter

    def _criteria_met(self, agent):
        return getattr(agent, self.counter) >= self.flag

class MaxTime(Terminator):
    def __init__(self, max_time):
        super().__init__()
        self.max_time = max_time
        self.start_time = time.time()

    def _criteria_met(self):
        current_time = time.time()
        return current_time >= self.start_time + self.max_time

