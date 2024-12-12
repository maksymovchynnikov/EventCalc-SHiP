# funcs/ship_setup.py

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Define constants
z_min = 32      # Minimum z-coordinate in meters
z_max = 82      # Maximum z-coordinate in meters
Delta_x_in = 1  # Delta x at z_min in meters
Delta_x_out = 4 # Delta x at z_max in meters
Delta_y_in = 2.7 # Delta y at z_min in meters
Delta_y_out = 6.2 # Delta y at z_max in meters

def x_max(z):
    """
    Calculate the maximum x-coordinate for a given z-coordinate.

    Parameters
    ----------
    z : float or np.ndarray
        Z-coordinate(s) in meters.

    Returns
    -------
    float or np.ndarray
        Maximum x-coordinate(s) in meters.
    """
    return (Delta_x_in/2 * (z - z_max) / (z_min - z_max) +
            Delta_x_out/2 * (z - z_min) / (z_max - z_min))

def y_max(z):
    """
    Calculate the maximum y-coordinate for a given z-coordinate.

    Parameters
    ----------
    z : float or np.ndarray
        Z-coordinate(s) in meters.

    Returns
    -------
    float or np.ndarray
        Maximum y-coordinate(s) in meters.
    """
    return (Delta_y_in/2 * (z - z_max) / (z_min - z_max) +
            Delta_y_out/2 * (z - z_min) / (z_max - z_min))

# Calculate theta_max_dec_vol
theta_max_dec_vol = np.arctan(
    max(
        np.sqrt((Delta_y_in / 2)**2 + (Delta_x_in / 2)**2) / z_min,
        np.sqrt((Delta_y_out / 2)**2 + (Delta_x_out / 2)**2) / z_max
    )
)

def plot_decay_volume(ax):
    """
    Plots the decay volume geometry as a trapezoidal prism on the given Axes3D object.
    The decay region extends in z from z_min to z_max.
    In x and y, its width is z-dependent based on predefined Delta values.
    The decay volume is colored light gray with gray edges.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object to plot on.
    """
    # Calculate x and y boundaries at z_min and z_max using the defined functions
    # At z_min
    x_min_zmin = -x_max(z_min)
    x_max_zmin_val = x_max(z_min)
    y_min_zmin = -y_max(z_min)
    y_max_zmin_val = y_max(z_min)

    # At z_max
    x_min_zmax = -x_max(z_max)
    x_max_zmax_val = x_max(z_max)
    y_min_zmax = -y_max(z_max)
    y_max_zmax_val = y_max(z_max)

    # Define the 8 vertices of the trapezoidal prism
    vertices = [
        [x_min_zmin, y_min_zmin, z_min],         # vertex0
        [x_max_zmin_val, y_min_zmin, z_min],     # vertex1
        [x_max_zmin_val, y_max_zmin_val, z_min], # vertex2
        [x_min_zmin, y_max_zmin_val, z_min],     # vertex3
        [x_min_zmax, y_min_zmax, z_max],         # vertex4
        [x_max_zmax_val, y_min_zmax, z_max],     # vertex5
        [x_max_zmax_val, y_max_zmax_val, z_max], # vertex6
        [x_min_zmax, y_max_zmax_val, z_max]      # vertex7
    ]

    # Define the 12 edges of the prism
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]

    # Plot the edges
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, color='gray', linewidth=1)

    # Define the 6 faces of the prism
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[3], vertices[0], vertices[4], vertices[7]]   # Left face
    ]

    # Create a Poly3DCollection for the faces
    face_collection = Poly3DCollection(faces, linewidths=0.5, edgecolors='gray', alpha=0.3)
    face_collection.set_facecolor('lightgray')  # Light gray with transparency
    ax.add_collection3d(face_collection)

def plot_decay_volume_plotly(fig):
    """
    Adds the SHiP decay volume to the given Plotly figure.
    The decay volume is represented as a trapezoidal prism with visible boundaries.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to add the decay volume to.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated Plotly figure with the decay volume added.
    """
    # Calculate x and y boundaries at z_min and z_max using the defined functions
    # At z_min
    x_min_zmin = -x_max(z_min)
    x_max_zmin_val = x_max(z_min)
    y_min_zmin = -y_max(z_min)
    y_max_zmin_val = y_max(z_min)

    # At z_max
    x_min_zmax = -x_max(z_max)
    x_max_zmax_val = x_max(z_max)
    y_min_zmax = -y_max(z_max)
    y_max_zmax_val = y_max(z_max)

    # Define the 8 vertices of the trapezoidal prism
    vertices = [
        [x_min_zmin, y_min_zmin, z_min],         # vertex0
        [x_max_zmin_val, y_min_zmin, z_min],     # vertex1
        [x_max_zmin_val, y_max_zmin_val, z_min], # vertex2
        [x_min_zmin, y_max_zmin_val, z_min],     # vertex3
        [x_min_zmax, y_min_zmax, z_max],         # vertex4
        [x_max_zmax_val, y_min_zmax, z_max],     # vertex5
        [x_max_zmax_val, y_max_zmax_val, z_max], # vertex6
        [x_min_zmax, y_max_zmax_val, z_max]      # vertex7
    ]

    # Define the faces of the decay volume
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [1, 2, 6, 5],  # Right face
        [2, 3, 7, 6],  # Back face
        [3, 0, 4, 7]   # Left face
    ]

    # Create mesh for the decay volume
    for face in faces:
        fig.add_trace(go.Mesh3d(
            x=[vertices[i][0] for i in face],
            y=[vertices[i][1] for i in face],
            z=[vertices[i][2] for i in face],
            color='rgba(200, 200, 200, 0.5)',  # Light gray with transparency
            opacity=0.5,
            name='Decay Volume',
            showscale=False,
            hoverinfo='skip'
        ))

    # Define the edges for the wireframe
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]

    # Add edges as Scatter3d lines
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[vertices[edge[0]][0], vertices[edge[1]][0]],
            y=[vertices[edge[0]][1], vertices[edge[1]][1]],
            z=[vertices[edge[0]][2], vertices[edge[1]][2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    return fig

def visualize_decay_volume():
    """
    Utility function to visualize the decay volume using Matplotlib's 3D plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_decay_volume(ax)
    
    # Set labels
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    
    # Set limits based on z_min and z_max
    x_lim = max(abs(x_max(z_min)), abs(x_max(z_max))) + 1
    y_lim = max(abs(y_max(z_min)), abs(y_max(z_max))) + 1
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_zlim(z_min, z_max +5)
    
    ax.set_title('SHiP Decay Volume')
    plt.show()

# Example usage:
# Uncomment the following lines to visualize the decay volume when this module is run directly.
#if __name__ == "__main__":
#    visualize_decay_volume()
