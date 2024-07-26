import torch

pi = torch.tensor([0.1, 0.2, 0.0, 0.7], requires_grad=True)
q  = torch.tensor([0.1, 0.2, 2, 0.7], requires_grad=False)

alpha = 3

def log_alpha(pi, alpha): # log_beta(2-alpha)
    if alpha == 1:
        return torch.log(pi)
    else:
        return (torch.pow(pi, alpha - 1) - 1)/(alpha - 1)

def Tsallis_Entropy(pi, alpha):
        return - pi * log_alpha(pi, alpha)/alpha

# compute policy loss (equation 10)
linear_term  = pi * q
entropy_term = Tsallis_Entropy(pi, alpha)
p_loss = -(linear_term + entropy_term).sum()

# Compute the gradient of entropy with respect to the policy
p_loss.backward()

# Print the gradient of the policy tensor
print("Policy tensor:", pi)
print("Gradient of the policy tensor:", pi.grad)