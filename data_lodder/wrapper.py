import torch
import torch.nn.functional as F

class SACWrapper(torch.nn.Module):
    def __init__(self, racing_net):
        super().__init__()
        # Rename to q1_net for consistency
        self.q1_net = racing_net
        
        # Create second Q-network
        self.q2_net = racing_net.__class__(
            state_dim=(racing_net.conv[0].in_channels, 96, 96),
            action_dim=(2,)
        )

    def forward(self, x):
        return self.q1_net(x)

    def get_q_values(self, states, actions):
        value1, alpha1, beta1 = self.q1_net(states)
        value2, alpha2, beta2 = self.q2_net(states)
        
        # Q-value calculations
        q1 = value1 + torch.sum(alpha1 * actions, dim=-1, keepdim=True)
        q2 = value2 + torch.sum(beta2 * actions, dim=-1, keepdim=True)
        
        # Scale Q-values
        q1 = torch.tanh(q1) * 20.0
        q2 = torch.tanh(q2) * 20.0
        
        return q1, q2

    def sample_action(self, states):
        _, alpha, beta = self.q1_net(states)
        
        # Parameter conditioning
        epsilon = 1e-6
        alpha = F.softplus(alpha) + epsilon
        beta = F.softplus(beta) + epsilon
        
        # Shape and condition parameters
        alpha = alpha.squeeze().clamp(0.1, 10.0)
        beta = beta.squeeze().clamp(0.1, 10.0)
        
        try:
            dist = torch.distributions.Beta(alpha, beta)
            
            # Sample with temperature
            temperature = 0.1
            action = dist.rsample() * (1 - temperature) + dist.mean * temperature
            
            # Compute log prob
            log_prob = dist.log_prob(action.clamp(1e-6, 1-1e-6)).sum(-1, keepdim=True)
            
            return action, log_prob, dist.mean
            
        except Exception as e:
            print(f"Distribution sampling error: {e}")
            mean = alpha / (alpha + beta)
            return mean, torch.zeros_like(mean), mean