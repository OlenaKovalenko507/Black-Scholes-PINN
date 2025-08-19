# Black–Scholes Equation Solver Using PINNs

This project implements a **Physics-Informed Neural Network (PINN)** to approximate the solution of the **Black–Scholes partial differential equation** for European call options. The network learns the option price as a function of stock price `S` and time `t`, and its predictions are compared in real-time with the analytical solution.

---

## 📖 Black–Scholes Equation

The **Black–Scholes PDE** for a European call option \(V(S,t)\) is:

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

**Boundary and terminal conditions:**

- Terminal condition at maturity \(t=T\):  

$$
V(S,T) = \max(S-K, 0)
$$

- Boundary conditions for stock price:  

$$
V(0,t) = 0, \quad V(S \to \infty, t) = S
$$

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `S` | Current stock price |
| `K` | Strike price of the option |
| `T` | Time to maturity |
| `t` | Current time |
| `\sigma` | Volatility of the underlying asset |
| `r` | Risk-free interest rate |
| `V(S,t)` | Option price as a function of `S` and `t` |

---

## ⚡ Project Features

- Physics-Informed Neural Network (PINN) approximates the solution of the PDE.
- Enforces **terminal payoff** and **boundary conditions** during training.
- Uses **automatic differentiation** to compute PDE residuals.
- **Live plotting** of the neural network prediction vs. the exact Black–Scholes solution.
- Fully written in **PyTorch** for easy experimentation.
- Can generate a **GIF of the training process** for visualization.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy
- imageio (for GIF creation)

Install dependencies with:

```bash
pip install torch numpy matplotlib scipy imageio
```
## 🚀 How to Run

1. Clone the repository or download the code.

2. Open the Python file in an IDE or terminal.

Run the script: 
```bash 
python black_scholes_pinn.py
```
3. A live plot will appear showing the neural network’s approximation and the exact solution as it trains.

4. After training, a GIF of the network training process will be saved as training_animation.gif.

## 📊 Example Outpu: 
- Blue line: Neural network prediction 
$$ 
𝑉(𝑆,0) 
- Red dashed line: Analytical Black–Scholes solution
- Plot updates in real-time as the network learns.

## 🔍 References
1. Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

