import matplotlib.pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
import numpy as np

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.legend()
    plt.show()


def plot_translations(T_original, parameter_updates):
    learnables = np.array([p['learnables'] for p in parameter_updates])
    iterations = [p['learnables'][0] for p in parameter_updates]
    T = learnables[:, 1, 0, :]
    T_x = T[:, 0]
    T_y = T[:, 1]
    T_z = T[:, 2]

    t = T_original.detach().cpu().numpy()[0]

    tx = [t[0] for _ in range(len(iterations))]
    ty = [t[1] for _ in range(len(iterations))]
    tz = [t[2] for _ in range(len(iterations))]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot T_x
    axs[0].plot(T_x, 'r-', label="T x")
    axs[0].plot(tx, 'r--', label="T x (original)")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("T x")
    axs[0].set_title("Translation Component T x")
    axs[0].legend()

    # Plot T_y
    axs[1].plot(T_y, 'g-', label="T y")
    axs[1].plot(ty, 'g--', label="T y (original)")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("T y")
    axs[1].set_title("Translation Component T y")
    axs[1].legend()

    # Plot T_z
    axs[2].plot(T_z, 'b-', label="T z")
    axs[2].plot(tz, 'b--', label="T z (original)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("T z")
    axs[2].set_title("Translation Component T z")
    axs[2].legend()

    plt.suptitle("Translation Components")
    plt.tight_layout()
    plt.show()



def plot_axis_angles(R_original, parameter_updates):
    learnables = np.array([p['learnables'] for p in parameter_updates])
    iterations = [p['learnables'][0] for p in parameter_updates]
    AA = learnables[:, 0, 0, :]
    AA_x = AA[:, 0]
    AA_y = AA[:, 1]
    AA_z = AA[:, 2]

    AA = matrix_to_axis_angle(R_original)
    AA = AA.detach().cpu().numpy()[0]

    A1 = [AA[0] for _ in range(len(iterations))]
    A2 = [AA[1] for _ in range(len(iterations))]
    A3 = [AA[2] for _ in range(len(iterations))]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot AA_x
    axs[0].plot(AA_x, 'r-', label="AA1")
    axs[0].plot(A1, 'r--', label="AA1 (original)")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("AA1")
    axs[0].set_title("Axis Angle Component 1")
    axs[0].legend()

    # Plot AA_y
    axs[1].plot(AA_y, 'g-', label="AA2")
    axs[1].plot(A2, 'g--', label="AA2 (original)")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("AA2")
    axs[1].set_title("Axis Angle Component 2")
    axs[1].legend()

    # Plot AA_z
    axs[2].plot(AA_z, 'b-', label="AA3")
    axs[2].plot(A3, 'b--', label="AA3 (original)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("AA3")
    axs[2].set_title("Axis Angle Component 3")
    axs[2].legend()

    plt.suptitle("Axis Angle Components")
    plt.tight_layout()
    plt.show()

