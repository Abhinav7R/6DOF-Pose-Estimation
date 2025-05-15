import torch
from math import radians, cos, sin

def initialize_R_T(R_gt, T_gt, rot_x, rot_y, rot_z, delx, dely, delz):
    """
    Initialize R_init and T_init based on given parameters.

    Args:
        R_gt (torch.Tensor): Ground truth rotation matrix of shape (1, 3, 3).
        T_gt (torch.Tensor): Ground truth translation vector of shape (1, 3).
        rot_x (float): Rotation in degrees around the x-axis.
        rot_y (float): Rotation in degrees around the y-axis.
        rot_z (float): Rotation in degrees around the z-axis.
        delx (float): Translation delta along the x-axis.
        dely (float): Translation delta along the y-axis.
        delz (float): Translation delta along the z-axis.

    Returns:
        R_init (torch.Tensor): Initialized rotation matrix of shape (1, 3, 3).
        T_init (torch.Tensor): Initialized translation vector of shape (1, 3).
    """
    # Convert rotation angles from degrees to radians
    rx, ry, rz = radians(rot_x), radians(rot_y), radians(rot_z)
    
    # Define rotation matrices for x, y, z axes
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos(rx), -sin(rx)],
                       [0, sin(rx), cos(rx)]])
    Ry = torch.tensor([[cos(ry), 0, sin(ry)],
                       [0, 1, 0],
                       [-sin(ry), 0, cos(ry)]])
    Rz = torch.tensor([[cos(rz), -sin(rz), 0],
                       [sin(rz), cos(rz), 0],
                       [0, 0, 1]])
    
    # Combine rotations: R_combined = Rz * Ry * Rx
    R_combined = Rz @ Ry @ Rx
    
    # Compute R_init by applying R_combined to R_gt
    R_init = R_combined @ R_gt.squeeze(0)
    R_init = R_init.unsqueeze(0)  # Restore batch dimension (1, 3, 3)
    
    # Compute T_init by adding translation deltas to T_gt
    T_init = T_gt + torch.tensor([[delx, dely, delz]])
    
    return R_init, T_init
