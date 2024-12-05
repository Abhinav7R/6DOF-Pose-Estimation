import torch
import numpy as np
import tqdm
import imageio
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mesh(path):
    verts, faces_idx, _ = load_obj(path)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh_ = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    # Scale the mesh
    scale = 0.5
    mesh_.scale_verts_(scale)
    return mesh_


def get_renderers(image_size, device):
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


def get_ground_truth(mesh, silhouette_renderer, phong_renderer, params):
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

    # Plotting the images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette_image, cmap='gray')
    plt.axis("off")
    plt.title("Silhouette")

    plt.subplot(1, 2, 2)
    plt.imshow(phong_image)
    plt.axis("off")
    plt.title("Phong")
    plt.show()

    return silhouette_image, phong_image, R, T


class AngleAxisModel(nn.Module):
    def __init__(self, meshes, renderer, ground_truth, params):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        distance = params['dist']
        elevation = params['elev']
        azimuth = params['azim']

        dist_var = params.get('dist_var', 10)
        elev_var = params.get('elev_var', 5)
        azim_var = params.get('azim_var', 0)

        ground_truth = torch.from_numpy((ground_truth != 0).astype(np.float32)).to(self.device)
        self.register_buffer('ground_truth', ground_truth)

        R, T = look_at_view_transform(
            dist=distance + torch.randn(1, device=device) * dist_var,
            elev=elevation + torch.randn(1, device=device) * elev_var,
            azim=azimuth + torch.randn(1, device=device) * azim_var,
            device=device)

        axis_angle = matrix_to_axis_angle(R)
        T += torch.randn(3, device=device) * 0.5
        self.AA = nn.Parameter(axis_angle)
        self.T = nn.Parameter(T)

    def forward(self):
        AA = self.AA
        R = axis_angle_to_matrix(AA)
        T = self.T

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.ground_truth) ** 2)
        return loss, image

n_epochs = 500

params = {
    "dist": 1000,
    "elev": -50,
    "azim": 50
}


mesh_path = "plane/plane.obj"
obj_name = mesh_path.split("/")[1].split(".")[0]
mesh = load_mesh(mesh_path)
image_size = 256
silhouette_renderer, phong_renderer = get_renderers(image_size, device)
ground_truth_image, _, R_actual, T_actual = get_ground_truth(mesh, silhouette_renderer, phong_renderer, params)



def train_angle_axis_model(mesh, ground_truth, silhouette_renderer, phong_renderer, object_name, n_epochs=500, params=None):
    filename_output = f"./{object_name}.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.5)

    # Initialize a model using the renderer, mesh and reference image
    model = AngleAxisModel(meshes=mesh, renderer=silhouette_renderer, ground_truth=ground_truth, params=params).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    plt.figure(figsize=(10, 10))

    _, image_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
    plt.grid(False)
    plt.title("Initial guess")
    plt.subplot(1, 2, 2)
    plt.imshow(model.ground_truth.cpu().numpy())
    plt.grid(False)
    plt.title("Ground truth")

    prev_loss = torch.tensor(0)
    losses = []
    parameter_updates = []

    R_min, T_min = None, None
    min_loss = float('inf')

    for i in tqdm.tqdm(range(n_epochs)):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
        
        tqdm.tqdm.write(f'iteration: {i}, loss: {loss}')

        losses.append(loss.item())
        parameter_updates.append({
            'iteration': i,
            'learnables': np.array([p.data.cpu().numpy() for p in model.parameters()]),
            # 'AA':  model.AA.clone().detach().cpu(),
            # 'T': model.T.clone().detach().cpu(),
            'loss': loss.item()
        })

        if loss < min_loss:
            R_min = model.AA
            T_min = model.T
            min_loss = loss

        if loss.item() < 200:
            break

        residue = torch.abs(loss - prev_loss)
        if residue < 100:
            residue = 0.001 * residue
            # add perturbation to R and T
            AA = model.AA
            T = model.T
            perturbation = torch.randn_like(AA) * residue
            AA = AA + perturbation
            dT = torch.randn_like(T) * residue
            T = T + dT
            model.AA.data = AA
            model.T.data = T

        if i % (n_epochs / 20) == 0:
            AA = model.AA
            R = axis_angle_to_matrix(AA)
            T = model.T

            image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
            
            plt.figure()
            plt.imshow(image[..., :3], cmap='viridis')
            plt.imshow(model.ground_truth.cpu().numpy().squeeze(), cmap='Reds', alpha=0.2)
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.axis("off")
            plt.show()

        prev_loss = loss

    writer.close()


    return losses, parameter_updates, R_min, T_min

losses, parameter_updates, R_pred, T_pred = train_angle_axis_model(mesh, ground_truth_image, silhouette_renderer, phong_renderer, obj_name, n_epochs, params)


# # Plot the losses
# plt.plot(losses)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.title('Loss vs iterations')
# plt.show()

# # Plot the updates of the parameters
# learnables = np.array([p['learnables'] for p in parameter_updates])
# AA = learnables[:, 0, 0, :]
# T = learnables[:, 1, 0, :]
# losses = [p['loss'] for p in parameter_updates]

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.plot(AA)
# plt.xlabel('Iteration')
# plt.ylabel('Angle Axis')
# plt.title('Angle Axis updates')
# plt.subplot(1, 2, 2)
# plt.plot(T)
# plt.xlabel('Iteration')
# plt.ylabel('Translation')
# plt.title('Translation updates')
# plt.show()

# Compare the predicted and actual R and T by rendering the mesh using the predicted R and T
R_pred = axis_angle_to_matrix(R_pred)
image_pred = phong_renderer(meshes_world=mesh, R=R_pred, T=T_pred)
image_pred = image_pred[0, ..., :3].detach().squeeze().cpu().numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_pred)
plt.axis("off")
plt.title("Predicted")

plt.subplot(1, 2, 2)
plt.imshow(ground_truth_image)
plt.axis("off")
plt.title("Actual")

plt.show()
