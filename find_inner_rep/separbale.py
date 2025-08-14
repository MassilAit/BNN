import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define {-1,1} space
points = np.array([[-1, -1],
                   [-1,  1],
                   [ 1, -1],
                   [ 1,  1]])

# Mapping: True=1, False=-1
def and_gate(x, y):
    return 1 if x == 1 and y == 1 else -1

def or_gate(x, y):
    return 1 if x == 1 or y == 1 else -1

# Meshgrid for shading
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300),
                     np.linspace(-1.5, 1.5, 300))

# Decision functions for {-1,1} encoding
zz_and = np.where(xx + yy > 1.5, 1, -1)
zz_or  = np.where(xx + yy > -0.5, 1, -1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Define colors and legend
colors = ['#ff9999', '#99ccff']  # -1, +1
legend_elements = [
    Patch(facecolor='#99ccff', edgecolor='k', label='Output +1'),
    Patch(facecolor='#ff9999', edgecolor='k', label='Output -1')
]

# Plot AND gate region
axes[0].contourf(xx, yy, zz_and, levels=[-2, 0, 2], colors=colors, alpha=0.6)
axes[0].contour(xx, yy, xx + yy - 1.5, levels=[0], colors='black', linewidths=2)
axes[0].scatter(points[:,0], points[:,1], c='black', s=80, marker='o')
axes[0].set_title("AND Gate")
axes[0].set_xticks([-1, 1])
axes[0].set_yticks([-1, 1])
axes[0].set_xlabel("Input A")
axes[0].set_ylabel("Input B")
axes[0].legend(handles=legend_elements, loc='upper left')
axes[0].grid(True)

# Plot OR gate region
axes[1].contourf(xx, yy, zz_or, levels=[-2, 0, 2], colors=colors, alpha=0.6)
axes[1].contour(xx, yy, xx + yy - (-0.5), levels=[0], colors='black', linewidths=2)
axes[1].scatter(points[:,0], points[:,1], c='black', s=80, marker='o')
axes[1].set_title("OR Gate")
axes[1].set_xticks([-1, 1])
axes[1].set_yticks([-1, 1])
axes[1].set_xlabel("Input A")
axes[1].set_ylabel("Input B")
axes[1].legend(handles=legend_elements, loc='upper left')
axes[1].grid(True)

plt.tight_layout()
plt.show()
