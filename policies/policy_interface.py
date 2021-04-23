from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class AbstractExpertModePolicy(ABC):
    @abstractmethod
    def update(self, cost_vector: np.ndarray):
        raise NotImplementedError

    @abstractproperty
    def policy(self):
        raise NotImplementedError

    @abstractmethod
    def act(self):
        raise NotImplementedError