import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# Step 1: Inputs and XOR labels
inputs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
labels = np.array([x ^ y ^ z for x, y, z in (inputs + 1) // 2])
colors = ["red" if label else "blue" for label in labels]

# Step 2: 3D scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=inputs[:, 0],
    y=inputs[:, 1],
    z=inputs[:, 2],
    mode='markers',
    marker=dict(size=6, color=colors),
    name="Input Points"
))

# Step 3: Plot neuron planes (example)
def plot_plane(fig, normal, name, size=1):
    d = 0  # No bias means plane goes through origin
    xx, yy = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    if normal[2] != 0:
        zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    else:
        zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, showscale=False, opacity=0.3, name=name))

# Example neuron planes (replace with real weights if needed)
example_weights = [
    np.array([-0.0106,  0.0014,  0.0098]),
    np.array([-0.0047, -0.0075, -0.0035]),
    np.array([ 0.0113, -0.0083,  0.0030])
]

for i, w in enumerate(example_weights):
    plot_plane(fig, w, name=f"Neuron {i+1}")

# Step 4: Add black axis lines manually
def add_axis_line(fig, axis):
    if axis == 'x':
        fig.add_trace(go.Scatter3d(
            x=[-1, 1], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))
    elif axis == 'y':
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-1, 1], z=[0, 0],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))
    elif axis == 'z':
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1, 1],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))

for axis in ['x', 'y', 'z']:
    add_axis_line(fig, axis)

# Layout settings with fixed cube aspect and grid/lines visible
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        yaxis=dict(title='Y', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        zaxis=dict(title='Z', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        aspectmode='cube'
    ),
    title="XOR 3D Inputs and Neuron Planes with Axes"
)

fig.show()


# Output layer input (from hidden layer) and their labels
points = np.array([
    [-1,  1, -1],
    [ 1,  1, -1],
    [ 1, -1, -1],
    [-1,  1,  1],
    [-1, -1,  1],
    [ 1, -1,  1]
])
labels = np.array([-1, 1, -1, 1, -1, 1])
colors = ["red" if label == 1 else "blue" for label in labels]

# Output layer weights (normal of decision plane)
output_plane_weight = np.array([0.0072, 0.0131, 0.0114])  # Replace with your real weights

# Create figure
fig = go.Figure()

# Add 3D scatter plot
fig.add_trace(go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=6, color=colors),
    name="Hidden Activations"
))

# Add decision plane
def plot_plane(fig, normal, name, size=1):
    d = 0  # No bias
    xx, yy = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    if normal[2] != 0:
        zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    else:
        zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, showscale=False, opacity=0.3, name=name))

plot_plane(fig, output_plane_weight, name="Output Plane")

# Add black coordinate axes
def add_axis_line(fig, axis):
    if axis == 'x':
        fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=4)))
    elif axis == 'y':
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='black', width=4)))
    elif axis == 'z':
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='black', width=4)))

for axis in ['x', 'y', 'z']:
    add_axis_line(fig, axis)

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Hidden 1', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        yaxis=dict(title='Hidden 2', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        zaxis=dict(title='Hidden 3', range=[-1, 1], showgrid=True, zeroline=True, showline=True),
        aspectmode='cube'
    ),
    title="Output Neuron Decision Plane in Hidden Activation Space"
)

fig.show()
