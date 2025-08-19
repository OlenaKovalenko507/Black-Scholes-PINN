import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------------
# Black-Scholes analytical solution (European Call)
# -------------------------------
def bs_call_price(S, K, T, r, sigma):
    """
    Compute the exact Black-Scholes European call option price.
    
    Parameters:
    S: array of stock prices
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: volatility
    
    Returns:
    V: array of option prices
    """
    S = np.array(S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# -------------------------------
# Neural Network Definition
# -------------------------------
class BlackScholesNN(nn.Module):
    def __init__(self, layers):
        """
        Fully-connected feedforward neural network for approximating V(S,t)
        
        layers: list of integers specifying neurons per layer
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            # Create linear layers
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()  # Nonlinear activation function
    
    def forward(self, x):
        # Forward pass through network
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)  # Output layer (linear)
        return x

# -------------------------------
# Physics-Informed Loss (PDE Residual)
# -------------------------------
def bs_pde_loss(model, S, t, sigma, r):
    """
    Compute the mean squared residual of the Black-Scholes PDE.
    
    model: neural network approximating V(S,t)
    S, t: training points (requires_grad=True for autodiff)
    sigma, r: Black-Scholes parameters
    """
    S.requires_grad = True  # needed for autograd
    t.requires_grad = True
    V = model(torch.cat([S, t], dim=1))  # concatenate S and t as input
    
    # Compute derivatives using automatic differentiation
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    
    # Black-Scholes PDE residual
    residual = V_t + 0.5*sigma**2 * S**2 * V_SS + r*S*V_S - r*V
    return torch.mean(residual**2)  # mean squared residual

# -------------------------------
# Generate Random Training Points
# -------------------------------
def generate_training_data(S_max, T, N):
    """
    Sample N random points (S,t) within the domain for training.
    
    S_max: maximum stock price
    T: maturity
    N: number of points
    """
    S = torch.rand(N,1)*S_max  # random S in [0, S_max]
    t = torch.rand(N,1)*T      # random t in [0, T]
    return S, t

# -------------------------------
# Terminal Payoff Loss
# -------------------------------
def terminal_loss(model, S, K, T):
    """
    Mean squared error of terminal condition V(S,T) = max(S-K,0)
    """
    t = torch.ones_like(S) * T  # t=T for all points
    V = model(torch.cat([S, t], dim=1))
    payoff = torch.maximum(S-K, torch.zeros_like(S))
    return torch.mean((V - payoff)**2)

# -------------------------------
# Boundary Loss
# -------------------------------
def boundary_loss(model, S, t, S_max):
    """
    Enforce boundary conditions:
    V(0,t) = 0, V(S_max,t) = S_max
    """
    V0 = model(torch.cat([torch.zeros_like(S), t], dim=1))  # S=0
    Vmax = model(torch.cat([torch.ones_like(S)*S_max, t], dim=1))  # S=S_max
    return torch.mean(V0**2) + torch.mean((Vmax - S_max)**2)

# -------------------------------
# Training Loop with Live Plot
# -------------------------------
sigma = 0.2
r = 0.05
K = 50
T = 1.0
S_max = 100

# Initialize neural network
model = BlackScholesNN([2, 32, 32, 32, 1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Stock prices to evaluate network predictions
S_plot = torch.linspace(0.01, S_max, 100).unsqueeze(1)  # avoid log(0)
t_plot = torch.zeros_like(S_plot)  # evaluate at t=0

# Exact solution for comparison
V_exact = bs_call_price(S_plot.numpy(), K, T, r, sigma)

# Set up live plot
plt.ion()
fig, ax = plt.subplots()
line_nn, = ax.plot([], [], 'b-', label='NN prediction')
line_exact, = ax.plot(S_plot.numpy(), V_exact, 'r--', label='Exact solution')
ax.set_xlim(0, S_max)
ax.set_ylim(0, S_max)
ax.set_xlabel('Stock Price S')
ax.set_ylabel('Option Price V(S,0)')
ax.set_title('Black-Scholes Call Option Approximation')
ax.legend()

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()  # reset gradients
    
    # Generate random training points
    S_train, t_train = generate_training_data(S_max, T, 1024)
    
    # Compute losses
    loss_pde = bs_pde_loss(model, S_train, t_train, sigma, r)
    loss_term = terminal_loss(model, S_train, K, T)
    loss_bound = boundary_loss(model, S_train, t_train, S_max)
    
    loss = loss_pde + loss_term + loss_bound  # total loss
    loss.backward()  # backpropagate
    optimizer.step()  # update weights
    
    # Update plot every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        with torch.no_grad():
            V_pred = model(torch.cat([S_plot, t_plot], dim=1)).numpy()
        line_nn.set_data(S_plot.numpy(), V_pred)
        plt.pause(0.01)

plt.ioff()
plt.show()
