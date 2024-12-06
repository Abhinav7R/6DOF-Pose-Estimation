import torch
import numpy as np
import tqdm
import imageio
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

class QuaternionModel(nn.Module):
    def __init__(self, mesh, renderer, gt_image, R_init, T_init, device):
        super().__init__()
        self.mesh = mesh
        self.renderer = renderer
        self.device = device

        gt_image = torch.from_numpy((gt_image != 0).astype(np.float32)).to(self.device)
        self.register_buffer('gt_image', gt_image)
        
        # Convert rotation matrix to quaternion
        quaternion = matrix_to_quaternion(R_init)
        self.quat = nn.Parameter(quaternion)
        self.T = nn.Parameter(T_init)
    
    @staticmethod
    def normalize_quaternion(q):
        """Normalize quaternion ensuring numerical stability"""
        norm = torch.sqrt(torch.sum(q * q) + 1e-8)
        return q / norm

    def forward(self):
        # Normalize quaternion to ensure valid rotation
        quat_normalized = self.normalize_quaternion(self.quat)
        R = quaternion_to_matrix(quat_normalized)
        T = self.T

        image = self.renderer(meshes_world=self.mesh.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.gt_image) ** 2)
        return loss, image

def train_quaternion_model(mesh, gt_image, R_init, T_init, silhouette_renderer, phong_renderer, object_name, n_epochs=500, device="cuda:0"):
    filename_output = f"./results/{object_name}_quaternion.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.5)

    # Initialize model
    model = QuaternionModel(mesh, silhouette_renderer, gt_image, R_init, T_init, device)
    
    # Use slightly lower learning rate for quaternions
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    plt.figure(figsize=(10, 10))
    
    _, image_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3], cmap='gray')
    plt.grid(False)
    plt.title("Initial guess")
    
    plt.subplot(1, 2, 2)
    plt.imshow(model.gt_image.cpu().numpy(), cmap='Reds')
    plt.grid(False)
    plt.title("Ground truth")
    plt.show()

    prev_loss = torch.tensor(0)
    losses = []
    parameter_updates = []

    Q_min, T_min = None, None
    min_loss = float('inf')

    for i in tqdm.tqdm(range(n_epochs)):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
        
        # Normalize quaternion after optimization step
        with torch.no_grad():
            model.quat.data = model.normalize_quaternion(model.quat.data)
        
        tqdm.tqdm.write(f'iteration: {i}, loss: {loss}')

        losses.append(loss.item())
        parameter_updates.append({
            'iteration': i,
            # 'learnables': np.array([p.data.cpu().numpy() for p in model.parameters()]),
            'quat': model.quat.clone().detach().cpu(),
            'T': model.T.clone().detach().cpu(),
            'loss': loss.item()
        })

        if loss < min_loss:
            Q_min = model.quat.clone()
            T_min = model.T.clone()
            min_loss = loss

        if loss.item() < 200:
            break

        residue = torch.abs(loss - prev_loss)
        if residue < 100:
            residue = 0.001 * residue
            # Add perturbation to quaternion and translation
            quat = model.quat
            T = model.T
            perturbation = torch.randn_like(quat) * residue
            quat = model.normalize_quaternion(quat + perturbation)
            dT = torch.randn_like(T) * residue
            T = T + dT
            model.quat.data = quat
            model.T.data = T

        if i % 10 == 0:
            quat = model.normalize_quaternion(model.quat)
            R = quaternion_to_matrix(quat)
            T = model.T

            image = phong_renderer(meshes_world=model.mesh.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            
            gt_image = model.gt_image.cpu().numpy().squeeze()
            gt_image_normalized = (gt_image - np.min(gt_image)) / (np.max(gt_image) - np.min(gt_image))
            gt_image_colored = np.stack([gt_image_normalized] * 3, axis=-1)

            # Blend the images
            alpha = 0.2  # Transparency for ground truth
            blended_image = (1 - alpha) * image + alpha * gt_image_colored

            # Convert to uint8 for GIF
            blended_image_ubyte = img_as_ubyte(blended_image)
            writer.append_data(blended_image_ubyte)

            # image = img_as_ubyte(image)
            # writer.append_data(image)
            
            plt.figure()
            plt.imshow(image[..., :3], cmap='viridis')
            plt.imshow(model.gt_image.cpu().numpy().squeeze(), cmap='Reds', alpha=0.2)
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.axis("off")
            plt.show()

        prev_loss = loss

    writer.close()

    return losses, parameter_updates, Q_min, T_min