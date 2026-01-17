import torch
import torch.nn as nn

class LorenzAttractor(nn.Module):
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        super().__init__()
        self.sigma = torch.tensor(sigma)
        self.rho = torch.tensor(rho)
        self.beta = torch.tensor(beta)

    def forward(self, x, y, z):
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return dx, dy, dz

# Initialize the model
model = LorenzAttractor()

# Random initial values for x, y, z
x = torch.rand(1)
y = torch.rand(1)
z = torch.rand(1)

# Compute dx, dy, dz
dx, dy, dz = model(x, y, z)

print(f"dx: {dx.item()}, dy: {dy.item()}, dz: {dz.item()}")