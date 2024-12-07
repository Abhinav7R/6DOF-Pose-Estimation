import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.legend()
    plt.show()

def plot_translations(T_original, parameter_updates):
    T = np.stack([p['T'] for p in parameter_updates])
    T_x = T[:, :, 0].flatten()
    T_y = T[:, :, 1].flatten()
    T_z = T[:, :, 2].flatten()


    t = T_original.detach().cpu().numpy().flatten()
    tx = [t[0] for _ in range(len(T))]
    ty = [t[1] for _ in range(len(T))]
    tz = [t[2] for _ in range(len(T))]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(T_x, 'r-', label="T x")
    axs[0].plot(tx, 'r--', label="T x (original)")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("T x")
    axs[0].set_title("Translation Component T x")
    axs[0].legend()

    axs[1].plot(T_y, 'g-', label="T y")
    axs[1].plot(ty, 'g--', label="T y (original)")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("T y")
    axs[1].set_title("Translation Component T y")
    axs[1].legend()

    axs[2].plot(T_z, 'b-', label="T z")
    axs[2].plot(tz, 'b--', label="T z (original)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("T z")
    axs[2].set_title("Translation Component T z")
    axs[2].legend()

    plt.suptitle("Translation Components")
    plt.tight_layout()
    plt.show()

def plot_quaternions(R_original, parameter_updates):
    Q = np.stack([p['quat'].numpy() for p in parameter_updates])
    Q_w = Q[:,:,0].flatten()  # real part
    Q_x = Q[:,:,1].flatten()  # i component
    Q_y = Q[:,:,2].flatten()  # j component
    Q_z = Q[:,:,3].flatten()  # k component

    # Get original quaternion
    Q_orig = matrix_to_quaternion(R_original)
    Q_orig = Q_orig.detach().cpu().numpy().flatten()

    # Create reference lines
    qw = [Q_orig[0] for _ in range(len(Q))]
    qx = [Q_orig[1] for _ in range(len(Q))]
    qy = [Q_orig[2] for _ in range(len(Q))]
    qz = [Q_orig[3] for _ in range(len(Q))]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Plot quaternion components
    axs[0, 0].plot(Q_w, 'k-', label="w")
    axs[0, 0].plot(qw, 'k--', label="w (original)")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("w")
    axs[0, 0].set_title("Quaternion Real Component (w)")
    axs[0, 0].legend()

    axs[0, 1].plot(Q_x, 'r-', label="x")
    axs[0, 1].plot(qx, 'r--', label="x (original)")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("x")
    axs[0, 1].set_title("Quaternion i Component (x)")
    axs[0, 1].legend()

    axs[1, 0].plot(Q_y, 'g-', label="y")
    axs[1, 0].plot(qy, 'g--', label="y (original)")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].set_title("Quaternion j Component (y)")
    axs[1, 0].legend()

    axs[1, 1].plot(Q_z, 'b-', label="z")
    axs[1, 1].plot(qz, 'b--', label="z (original)")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("z")
    axs[1, 1].set_title("Quaternion k Component (z)")
    axs[1, 1].legend()

    plt.suptitle("Quaternion Components")
    plt.tight_layout()
    plt.show()

    # Plot quaternion norm over iterations
    plt.figure(figsize=(10, 5))
    norms = np.sqrt(Q_w**2 + Q_x**2 + Q_y**2 + Q_z**2)
    plt.plot(norms, label="Quaternion Norm")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Unit Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.title("Quaternion Norm Over Iterations")
    plt.legend()
    plt.show()