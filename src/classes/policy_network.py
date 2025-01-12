import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device="cpu", hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.device = device

    def forward(self, state, t):
        t = torch.tensor([t] * len(state)).to(self.device)
        t = t.unsqueeze(1)
        state = torch.cat([state, t], dim=-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std

    # def __init__(self, state_dim, action_dim, device='cpu', hidden_dim=128):
    #     super(PolicyNetwork, self).__init__()
    #     self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.mean_layer = nn.Linear(hidden_dim, action_dim)
    #     self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    #     self.device = device

    #     # Initialize weights
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for layer in [self.fc1, self.fc2, self.mean_layer, self.log_std_layer]:
    #         nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    #         nn.init.zeros_(layer.bias)

    # def forward(self, state, t):
    #     # Ensure input is on the correct device
    #     state = state.to(self.device)
    #     t = torch.full((state.size(0), 1), t, device=self.device)  # Create a tensor with time t
    #     state_with_time = torch.cat([state, t], dim=-1)

    #     # Forward pass
    #     x = F.relu(self.fc1(state_with_time))
    #     x = F.relu(self.fc2(x))
    #     mean = self.mean_layer(x)
    #     log_std = self.log_std_layer(x)
    #     std = torch.exp(log_std)  # Ensure std is always positive

    #     return mean, std
