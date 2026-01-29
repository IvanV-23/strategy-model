import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim_diplomacy, action_dim_economy):
        super(ActorCritic, self).__init__()
        
        # 1. The Shared Backbone (The "Understanding" of the game)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 2. Action Head: Diplomacy (Trade, Pass, Attack)
        self.diplomacy_head = nn.Linear(128, action_dim_diplomacy)
        
        # 3. Action Head: Economy (Invest, Create Units, Idle)
        self.economy_head = nn.Linear(128, action_dim_economy)
        
        # 4. The Critic Head: Evaluates how "good" the current state is
        # This helps the agent learn if its combination of choices was smart.
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        # Process the input state through the shared backbone
        x = self.backbone(state)
        
        # Calculate probabilities for each action branch
        # We use Softmax so the output adds up to 1.0 (probabilities)
        diplomacy_logits = self.diplomacy_head(x)
        economy_logits = self.economy_head(x)
        
        # Calculate the state value (Critic)
        state_value = self.value_head(x)
        
        return diplomacy_logits, economy_logits, state_value

    def get_action(self, state):
        """Helper function to sample actions during gameplay."""
        diplomacy_logits, economy_logits, _ = self.forward(state)
        
        # Convert logits to probabilities
        prob_dip = F.softmax(diplomacy_logits, dim=-1)
        prob_eco = F.softmax(economy_logits, dim=-1)
        
        # Sample an action (exploration)
        dist_dip = torch.distributions.Categorical(prob_dip)
        dist_eco = torch.distributions.Categorical(prob_eco)
        
        return dist_dip.sample(), dist_eco.sample()
