import numpy as np
import torch as torch

class BlobAgent:
    def __init__(self, model, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = model
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.model.fc2.out_features))  # Random action (exploration)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()  # Action with highest Q-value (exploitation)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

# Later, after initializing the neural network model, instantiate the agent:
# blob_agent = BlobAgent(model)
