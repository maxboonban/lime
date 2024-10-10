import numpy as np
import plotly.graph_objects as go

# It is perfectly okay if you have no idea what's happening in this file.
# Most of this file is about creating nice looking 3D visualizations with plotly.

default_offset = 0.5
default_ground_level = -5.0

default_plasma_min = 0.0
default_plasma_max = 20.0

def visualize_data(X, y_true, 
                  cmin=default_plasma_min, cmax=default_plasma_max):
    fig_linreg = go.Figure()
    x1, x2, y = X[:, 0], X[:, 1], y_true.reshape(-1)
    go_linreg = go.Scatter3d(
        x=x1, y=x2, z=y,
        mode="markers",
        marker=dict(size=4, color=y, colorscale="Plasma", 
                    cmin=0, cmax=20, opacity=0.8)
    )

    fig_linreg.add_trace(go_linreg)
    fig_linreg.update_layout(
        width=600, height=400, showlegend=False,
        margin = dict(l=0, r=0, b=0, t=0), 
        scene = dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="y", 
            aspectratio= {"x":1.2, "y":1.2, "z":0.9}),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.2, y=1.8, z=0.))
    )

    return fig_linreg


def get_model_go(pred_function, 
                 cmin=default_plasma_min, cmax=default_plasma_max):
    """
    go = graphics object
    """
    x1_grid, x2_grid = np.linspace(-8, 8, 33), np.linspace(-8, 8, 33)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="xy")
    x3_mesh = np.zeros(x2_mesh.shape)
    y_pred_mesh = pred_function(x1_mesh, x2_mesh, x3_mesh)
    
    go_model = go.Surface(
        x=x1_mesh, y=x2_mesh, z=y_pred_mesh, 
        colorscale="Plasma", cmin=cmin, cmax=cmax, showscale=False, opacity=0.4)
    
    return go_model


def get_glass_go(obs_point, disp_radius, coef_glassbox, inter_glassbox):
    """
    go = graphics object
    """
    x1_obs, x2_obs = obs_point[0], obs_point[1]
    x1_grid = np.linspace(x1_obs-disp_radius, x1_obs+disp_radius, 3)
    x2_grid = np.linspace(x2_obs-disp_radius, x2_obs+disp_radius, 3)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="xy")
    
    y_glass_mesh = inter_glassbox + coef_glassbox[0]*x1_mesh + + coef_glassbox[1]*x2_mesh
    
    go_glass = go.Surface(
        x=x1_mesh, y=x2_mesh, z=y_glass_mesh, 
        colorscale=[[0, 'lime'], [1, 'lime']], showscale=False, opacity=0.4)
    
    return go_glass


def get_perturb_go(perturbed_samples, perturbed_predictions, perturbad_weights):
    """
    go = graphics object
    """
    x1_perturb, x2_perturb = perturbed_samples[:, 0], perturbed_samples[:, 1]
    y_perturb = perturbed_predictions.reshape(-1)
    
    size_scale = 5.0/np.mean(perturbad_weights)
    
    go_perturb = go.Scatter3d(
        x=x1_perturb, y=x2_perturb, z=y_perturb, 
        mode="markers",
        marker=dict(color="green", size=size_scale*perturbad_weights, opacity=1)
    )
    
    return go_perturb


def visualize_difficult_model(difficult_model):
    fig_difficult = go.Figure()
    x1_grid, x2_grid = np.linspace(-8, 8, 63), np.linspace(-8, 8, 63)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="xy")

    y_pred_mesh = difficult_model(x1_mesh, x2_mesh)

    go_model = go.Surface(
        x=x1_mesh, y=x2_mesh, z=y_pred_mesh, 
        colorscale="Plasma", cmin=-1.0, cmax=9.0, showscale=False, opacity=0.7)

    fig_difficult.add_trace(go_model)

    fig_difficult.update_layout(
        width=600, height=400, showlegend=False,
        margin = dict(l=0, r=0, b=0, t=0), 
        scene = dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="y", 
            aspectratio= {"x":1.2, "y":1.2, "z":0.9}),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.2, y=1.8, z=0.0))
    )
    
    return fig_difficult



