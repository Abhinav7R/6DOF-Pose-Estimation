"""
This file is used to start the 6dof pose estimation
It contains functions to load the object and the renderers
It returns the mesh of the object and the renderers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

def load_mesh(path, device="cuda:0", scale=None):
    """
    Load the mesh of the object and scale it
    """
    verts, faces_idx, _ = load_obj(path)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh_ = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    # Scale the mesh
    if scale is not None:
        mesh_.scale_verts_(scale)
    return mesh_

def get_renderers(image_size=256, device="cuda:0"):
    """
    Get the renderers for the object
    """
    # Initialize an OpenGL perspective camera.
    cameras = FoVPerspectiveCameras(device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    silhouette_raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100,
        # bin_size=0,
        # max_faces_per_bin=100000
    )

    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=silhouette_raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    phong_raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        # bin_size=0, 
        # max_faces_per_bin=100000
    )

    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=phong_raster_settings,
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return silhouette_renderer, phong_renderer


def get_ground_truth(mesh, silhouette_renderer, phong_renderer, params, device="cuda:0"):
    dist = params['dist']
    elev = params['elev']
    azim = params['azim']

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    R = R.to(device)
    T = T.to(device)

    # Render the silhouette of the mesh
    silhouette = silhouette_renderer(mesh, R=R, T=T)
    silhouette_image = silhouette[0, ..., 3].detach().cpu().numpy()  # Extract alpha channel

    # Render the Phong shaded image
    phong = phong_renderer(mesh, R=R, T=T)
    phong_image = phong[0, ..., :3].detach().cpu().numpy()  # Extract RGB channels

    return silhouette_image, phong_image, R, T

def plot_images(silhouette_image, phong_image, gt=True):
    """
    Plot the ground truth images
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(silhouette_image, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Silhouette")
    axs[1].imshow(phong_image)
    axs[1].axis("off")
    axs[1].set_title("Phong Shading")
    if gt:
        plt.suptitle("Ground Truth")
    else:
        plt.suptitle("Initial Guess")
    plt.show()


def get_initial_R_T(params, variances, deltaT=0.5, device="cuda:0"):
    """
    Get the initial rotation and translation
    """
    
    distance = params['dist']
    elevation = params['elev']
    azimuth = params['azim']

    dist_var = variances['dist']
    elev_var = variances['elev']
    azim_var = variances['azim']

    R_init, T_init = look_at_view_transform(
                dist=distance + torch.randn(1, device=device) * dist_var,
                elev=elevation + torch.randn(1, device=device) * elev_var,
                azim=azimuth + torch.randn(1, device=device) * azim_var,
                device=device)
    
    T_init += torch.randn(3, device=device) * deltaT

    return R_init, T_init

def plot_initial_guess(mesh, silhouette_renderer, phong_renderer, R_init, T_init):
    """
    Plot the initial guess
    """
    silhouette = silhouette_renderer(mesh, R=R_init, T=T_init)
    silhouette_image = silhouette[0, ..., 3].detach().cpu().numpy()  # Extract alpha channel

    phong = phong_renderer(mesh, R=R_init, T=T_init)
    phong_image = phong[0, ..., :3].detach().cpu().numpy()  # Extract RGB channels

    plot_images(silhouette_image, phong_image, gt=False)




