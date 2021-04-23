import numpy as np
from policies.policy_interface import AbstractExpertModePolicy


class ExpertModeHedge(AbstractExpertModePolicy):
    def __init__(self, num_actions: int, lr: float):
        self.n = num_actions
        # Initialise policy to the uniform distribution
        self._policy = np.ones(self.n) / self.n
        self.lr = lr
        self.policies_sum = np.zeros(self.n)
        self.time_avg_policy = self.policy
        self.count = 0

    def update(self, cost_vector: np.ndarray) -> None:
        assert cost_vector.shape == self.policy.shape, "Loss vector must be equal in shape to the policy."
        self.count += 1
        updated_values = self.policy * np.exp(-self.lr * cost_vector)
        # Renormalise so that the policy values sum to unity.
        self._policy = updated_values / np.sum(updated_values)
        self.policies_sum += self.policy
        self.time_avg_policy = self.policies_sum / self.count
        self.time_avg_policy /= np.sum(self.time_avg_policy)
        assert np.isclose(np.sum(self.time_avg_policy), 1.0)

    @property
    def policy(self):
        return self._policy

    def act(self):
        return np.random.choice(self.n, p=self.policy)


class ExpertModeMultiplicativeWeights(AbstractExpertModePolicy):
    def __init__(self, num_actions: int, epsilon: float):
        self.n = num_actions
        self._weights = np.ones(self.n)
        self._epsilon = epsilon
        self.policies_sum = np.zeros(self.n)
        self.time_avg_policy = self.policy
        self.count = 0

    def update(self, cost_vector: np.ndarray) -> None:
        assert cost_vector.shape == self.policy.shape, "Loss vector must be equal in shape to the policy."
        self.count += 1
        self._weights = self._weights * (1 - self._epsilon) ** cost_vector
        self.policies_sum += self.policy
        self.time_avg_policy = self.policies_sum / self.count
        self.time_avg_policy /= np.sum(self.time_avg_policy)
        assert np.isclose(np.sum(self.time_avg_policy), 1.0)

    @property
    def policy(self):
        return self._weights / self._weights.sum()

    def act(self):
        return np.random.choice(self.n, p=self.policy)

    def set_weights(self, new_weights):
        self._weights = new_weights

    def get_weights(self):
        return self._weights


class ExpertOptimisticModeMultiplicativeWeights(AbstractExpertModePolicy):
    def __init__(self, num_actions: int, lr: float):
        self.n = num_actions
        # Initialise policy to the uniform distribution
        self._policy = np.ones(self.n) / self.n
        self.lr = lr
        self.policies_sum = np.zeros(self.n)
        self.time_avg_policy = self.policy
        self.count = 0
        self.previous_cost_vector = np.zeros(self.n)

    def update(self, cost_vector: np.ndarray) -> None:
        assert cost_vector.shape == self.policy.shape, "Loss vector must be equal in shape to the policy."
        self.count += 1
        updated_values = self.policy * np.exp(-2*self.lr*cost_vector+self.lr*self.previous_cost_vector)
        self.previous_cost_vector = cost_vector
        self._policy = updated_values / np.sum(updated_values)
        self.policies_sum += self.policy
        self.time_avg_policy = self.policies_sum / self.count
        self.time_avg_policy /= np.sum(self.time_avg_policy)
        assert np.isclose(np.sum(self.time_avg_policy), 1.0)

    @property
    def policy(self):
        return self._policy

    def act(self):
        return np.random.choice(self.n, p=self.policy)


class ExpertModeFictitiousPlay(AbstractExpertModePolicy):
    def __init__(self, num_actions: int, epsilon: float):
        self.n = num_actions
        self.empirical_counts = np.ones(self.n)
        self.best_response = None
        self.time_avg_policy = self.policy

    def update(self, cost_vector: np.ndarray) -> None:
        self.best_response = np.argmin(cost_vector)
        self.empirical_counts[self.best_response] += 1
        self.time_avg_policy = self.policy

    @property
    def policy(self):
        return self.empirical_counts/self.empirical_counts.sum()

    def act(self):
        return np.random.choice(self.n, p=self.policy)
