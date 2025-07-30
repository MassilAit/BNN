import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-3, 3, 1000)

# Forward functions
sign_forward = np.sign(x)
tanh_forward = np.tanh(x)

# STE gradient approximation: clipped identity
ste_grad = (np.abs(x) <= 1).astype(float)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot STE (forward + gradient)
axs[1].plot(x, sign_forward, label="sign(x)", color='blue')
axs[1].plot(x, ste_grad, label="STE gradient (∂x)", linestyle='--', color='red')
axs[1].set_title("Sign Function with STE")
axs[1].legend()
axs[1].grid(True)

# Plot tanh
axs[0].plot(x, tanh_forward, label="tanh(x)", color='green')
axs[0].plot(x, 1 - np.tanh(x)**2, label="∂tanh/∂x", linestyle='--', color='orange')
axs[0].set_title("Tanh Function and Derivative")
axs[0].legend()
axs[0].grid(True)

plt.tight_layout()
plt.savefig("fig_ste_vs_tanh.png", dpi=300)
plt.show()
