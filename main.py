'''
Very simple implementation of the paper 'Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Di erential Equations'
by Raisi et.al on the 1D Heat equation with Dirchlet Boundary Conditions and a sine function as the initial condition
'''

import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

STEPS = 1500

# For the colocation points to calculate MSE_f
t = torch.linspace(start=0, end=1, steps=STEPS, requires_grad=True)
x = torch.linspace(start=0, end=1, steps=STEPS, requires_grad=True)


# Boundary conditions for the time
t_b = torch.linspace(start=0, end=1, steps=STEPS, requires_grad=True)

# Left and right boundary condition in this case using torch.ones because length of the rod is 1
x_0 = torch.zeros(STEPS, requires_grad=True)
x_L = torch.ones(STEPS, requires_grad=True)

u_0 = 7 * torch.sin(torch.pi * x) # Initial condition ground truth
t0 = torch.zeros(STEPS, requires_grad=True) # For the initial condition

class PINN(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 200),
        nn.Tanh(),
        nn.Linear(200, 100),
        nn.Tanh(),
        nn.Linear(100, 1)
    )

  def forward(self, x):
    return self.net(x)


pinn = PINN()

def u(t, x):
  u = pinn(torch.stack((t, x), dim=1))
  return u

def f(t, x, alpha=1):
  o = u(t, x)[:, 0]
  u_t = torch.autograd.grad(o, t, grad_outputs=torch.ones_like(o), create_graph=True)[0]
  u_x = torch.autograd.grad(o, x, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True)[0]
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True)[0]

  f = u_t - (alpha * u_xx)
  return f

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(pinn.parameters())
epochs = 100
losses = []

for i in range(epochs):
  optimizer.zero_grad()

  # Calculating MSE_u for the boundary conditions and the initial condition respectively

  # For the left and right boundary conditions
  u_left = u(t_b, x_0)
  left_bc_loss = loss(u_left, torch.zeros_like(u_left))
  u_right = u(t_b, x_L)
  right_bc_loss = loss(u_right, torch.zeros_like(u_right))

  # For the initial condition (t0, x)
  u_initial = u(t0, x)
  loss_initial = loss(u_initial, u_0)
  mse_u = left_bc_loss + right_bc_loss + loss_initial

  # Calculatingf MSE_f by evaluating it on the colocation points f(t, x)
  coloc_points = f(t, x, alpha=1)
  mse_f = torch.mean(coloc_points ** 2)

  total_loss = mse_u + mse_f
  losses.append(total_loss.item())
  print(f"Epoch: {i+1}, Loss: {total_loss.item()}")
  total_loss.backward(retain_graph=True)
  optimizer.step()


# Visualize the loss 
plt.plot(losses)
plt.title("Losses over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Total Loss")
plt.savefig("Loss")
plt.show()

# Testing the model
t_test = torch.linspace(0, 1, 100)  
x_test = torch.linspace(0, 1, 100)  
T, X = torch.meshgrid(t_test, x_test, indexing='ij')

# Flatten the grid for model input
t_flat = T.flatten()
x_flat = X.flatten()

# Evaluate the trained PINN model
with torch.no_grad():  
  u_pred = pinn(torch.stack((t_flat, x_flat), dim=1)).detach()

# Reshape predictions to match the grid
U = u_pred.view(100, 100).numpy()

# Plot the solution using a heatmap
plt.figure(figsize=(10, 6))
plt.imshow(U, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label="u(t, x)")
plt.title("1D Heat Equation Solution")
plt.xlabel("x")
plt.ylabel("t")
plt.savefig("Result")
plt.show()