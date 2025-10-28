import torch
batch_size =2 
seq_length = 5 
hidden_size = 1
gamma = 0.5

torch.manual_seed(42)

rewards = torch.randn((batch_size, seq_length))
values = torch.rand((batch_size, seq_length))

advantages = torch.zeros_like(rewards)
returns = torch.zeros_like(rewards)


last_gae_lam = 0
for t in range(seq_length-1, -1, -1):
    if t == seq_length-1:
        next_value = 0
    else:
        next_value = values[:, t+1]

    delta = rewards[:, t] + gamma * next_value - values[:, t]
    advantages[:, t] = last_gae_lam = delta + gamma * last_gae_lam

returns = advantages + values

print(advantages)
print(returns)


