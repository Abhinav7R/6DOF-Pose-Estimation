# Load the mesh

# In[0]:
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import time
from scipy import linalg
import cv2
import random

axis_0 = [0, 0, 1]
angle_0 = 10

axis_1 = [0, 0, 1]
angle_1 = 40



# In[1]:
car_mesh = o3d.io.read_triangle_mesh("car.obj")
car_mesh.compute_vertex_normals()


# In[3]:
def axis_angle_to_rotation_matrix(axis, angle_degrees):
    """
    Convert axis-angle to 3x3 rotation matrix
    
    Args:
        axis: 3D vector representing rotation axis
        angle_degrees: rotation angle in degrees
    Returns:
        3x3 rotation matrix
    """
    # Normalize axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Create rotation object and get matrix
    rot = Rotation.from_rotvec(axis * np.radians(angle_degrees))
    return rot.as_matrix()


def rotation_matrix_to_axis_angle(R_matrix):
    """
    Convert a 3x3 rotation matrix to axis-angle representation.
    
    Args:
        R_matrix (numpy.ndarray): 3x3 rotation matrix.
        
    Returns:
        axis (numpy.ndarray): 3D unit vector representing the axis of rotation.
        angle (float): Rotation angle in radians.
        angle_degrees (float): Rotation angle in degrees.
    """
    # Create a Rotation object from the rotation matrix
    rotation = Rotation.from_matrix(R_matrix)
    
    # Convert to rotation vector (axis-angle representation)
    rotvec = rotation.as_rotvec()
    
    # The direction of the rotation vector is the axis of rotation
    # The magnitude of the rotation vector is the angle of rotation in radians
    axis = rotvec / np.linalg.norm(rotvec)
    angle = np.linalg.norm(rotvec)
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    
    return axis, angle, angle_degrees

# In[4]:
# Define the camera extrinsics
extrinsics_1 = np.identity(4)
extrinsics_1[:3, :3] = axis_angle_to_rotation_matrix(axis_0, angle_0)
extrinsics_1[:3, 3] = [0, 0, 2]
print(rotation_matrix_to_axis_angle(extrinsics_1[:3, :3]))

extrinsics_2 = np.identity(4)
extrinsics_2[:3, :3] = axis_angle_to_rotation_matrix(axis_1, angle_1)
extrinsics_2[:3, 3] = [0,0,2]

# In[5]:
# Create the camera
params = o3d.camera.PinholeCameraParameters()
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
for idx ,camera_transform in enumerate([extrinsics_1, extrinsics_2]):
    vis.add_geometry(car_mesh)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params)
    ctr.set_lookat(camera_transform[:3, 3])
    ctr.set_front(camera_transform[:3, 2])
    ctr.set_up(camera_transform[:3, 1])
    print(params.extrinsic)
    print(params.intrinsic.intrinsic_matrix)
    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer()
    time.sleep(1)
    vis.remove_geometry(car_mesh)
    # vis.destroy_window()

    image_path = f"rendered_image_{idx}.png"
    o3d.io.write_image(image_path, o3d.geometry.Image((np.asarray(image) * 255).astype(np.uint8)))
vis.destroy_window()

# In[6]:
# Display the images
from PIL import Image
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Load images
image_0 = Image.open("rendered_image_0.png")
image_1 = Image.open("rendered_image_1.png")

# Display images
axes[0].imshow(image_0)
axes[0].set_title("Rendered Image 0")
axes[0].axis('off')

axes[1].imshow(image_1)
axes[1].set_title("Rendered Image 1")
axes[1].axis('off')

plt.show()

# In[7]:


def match_features(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
    matches_mask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),
                      singlePointColor=None,
                      matchesMask=matches_mask,
                      flags=2,
                      )

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB ))
    plt.title(f'Found {sum(matches_mask)} inlier matches out of {len(good_matches)} matches')
    plt.axis('off')
    plt.show()

    return H, good_matches, matches_mask, kp1, kp2

# H, matches, mask, kp1, kp2 = match_features('data_Q2/first_img.jpg', 'data_Q2/second_img.jpg')
H, matches, mask, kp1, kp2 = match_features('rendered_image_0.png', 'rendered_image_1.png')

# In[8]:
def get_inlier_points(matches, mask, kp1, kp2, num_points=8):
    inlier_matches = [m for m, inlier in zip(matches, mask) if inlier]
    
    if len(inlier_matches) < num_points:
        raise ValueError(f"Not enough inlier points. Found {len(inlier_matches)}, required {num_points}.")
    
    selected_matches = inlier_matches[:num_points]
    random.seed(42)  
    selected_matches = random.sample(inlier_matches, num_points)
    points1 = np.float32([kp1[m.queryIdx].pt for m in selected_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in selected_matches])
    
    return points1, points2

total_points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
total_points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
# Filter out outliers using matches mask
inlier_total_points1 = total_points1[np.array(mask) == 1]
inlier_total_points2 = total_points2[np.array(mask) == 1]
# print(inlier_total_points1)
print(inlier_total_points1.shape, inlier_total_points2.shape)
points1, points2 = get_inlier_points(matches, mask, kp1, kp2)
# print("Points in Image 1:", points1)
# print("Points in Image 2:", points2)

# In[9]:
def overlay_points(img1_path, img2_path, points1, points2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    for point in points1:
        cv2.circle(img1, (int(point[0]), int(point[1])), 30, (0, 255, 0), -1)
    
    for point in points2:
        cv2.circle(img2, (int(point[0]), int(point[1])), 30, (0, 255, 0), -1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Sampled Points from Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Corresponding Sampled Points from Image 2')
    plt.axis('off')
    
    plt.show()

# Example usage:
# overlay_points('data_Q2/first_img.jpg', 'data_Q2/second_img.jpg', points1, points2)
overlay_points('rendered_image_0.png', 'rendered_image_1.png', points1, points2)

# In[10]:
def normalize_points(points):
    """
    Normalize points to improve numerical stability
    Returns normalized points and transformation matrix
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Calculate average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(points_centered**2, axis=1)))
    scale = np.sqrt(2) / avg_dist
    
    # Create transformation matrix
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])
    
    # Apply transformation
    points_homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized_points = (T @ points_homogeneous.T).T[:, :2]
    
    return normalized_points, T


# In[11]:
def compute_fundamental_matrix_8point(points1, points2):
    """
    Compute fundamental matrix using normalized 8-point algorithm with best solution selection
    """
    # Normalize points
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)
    
    # Build constraint matrix
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = norm_points1[i]
        x2, y2 = norm_points2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve using SVD
    U, S, Vh = np.linalg.svd(A)
    
    # Get multiple possible solutions from last two singular vectors
    F1 = Vh[-1].reshape(3, 3)
    F2 = Vh[-2].reshape(3, 3)
    
    # Try different combinations
    best_F = None
    min_error = float('inf')
    
    for alpha in np.linspace(0, 1, 100000):  # Try different weighted combinations
        F = alpha * F1 + (1-alpha) * F2
        
        # Enforce rank 2
        U_f, S_f, Vh_f = np.linalg.svd(F)
        S_f[2] = 0
        F = U_f @ np.diag(S_f) @ Vh_f
        
        # Denormalize
        F = T2.T @ F @ T1
        
        # Normalize
        F = F / np.sqrt((F**2).sum())
        
        # Calculate error
        error = calculate_epipolar_error(points1, points2, F)
        
        if error < min_error:
            min_error = error
            best_F = F
    
    return best_F


def compute_fundamental_matrix_7point(points1, points2):
    """
    Compute fundamental matrix using 7-point algorithm
    """
    # Normalize points
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)
    
    # Build constraint matrix
    A = np.zeros((7, 9))
    for i in range(7):
        x1, y1 = norm_points1[i]
        x2, y2 = norm_points2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Find null space
    U, S, Vh = np.linalg.svd(A)
    F1 = Vh[-1].reshape(3, 3)
    F2 = Vh[-2].reshape(3, 3)
    
    # Solve cubic equation det(αF1 + (1-α)F2) = 0
    def det_F(alpha):
        F = alpha * F1 + (1-alpha) * F2
        return np.linalg.det(F)
    
    # Find roots of cubic equation (simplified version)
    coeffs = np.polynomial.polynomial.polyfit(np.linspace(0, 1, 100),
                                            [det_F(x) for x in np.linspace(0, 1, 100)], 3)
    alphas = np.roots(coeffs[::-1])
    
    # Find best solution
    best_F = None
    min_error = float('inf')
    
    for alpha in alphas.real[abs(alphas.imag) < 1e-4]:
        F = alpha * F1 + (1-alpha) * F2
        F = T2.T @ F @ T1
        F = F / F[2,2]
        
        # Calculate error
        error = calculate_epipolar_error(points1, points2, F)
        if error < min_error:
            min_error = error
            best_F = F
    
    return best_F

def calculate_epipolar_error(points1, points2, F):
    """
    Calculate average epipolar line distance error
    """
    points1_homog = np.column_stack((points1, np.ones(len(points1))))
    points2_homog = np.column_stack((points2, np.ones(len(points2))))
    
    # Calculate epipolar lines
    lines2 = points1_homog @ F.T
    lines1 = points2_homog @ F
    
    # Calculate distances
    dist1 = np.abs(np.sum(points1_homog * lines1, axis=1)) / \
            np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    dist2 = np.abs(np.sum(points2_homog * lines2, axis=1)) / \
            np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    return np.mean(dist1 + dist2)

def compare_methods(points1, points2):
    """
    Compare 8-point and 7-point algorithms
    """
    F_8point = compute_fundamental_matrix_8point(points1, points2)
    F_7point = compute_fundamental_matrix_7point(points1[:7], points2[:7])
    
    error_8point = calculate_epipolar_error(points1, points2, F_8point)
    error_7point = calculate_epipolar_error(points1, points2, F_7point)
    
    return {
        '8-point error': error_8point,
        '7-point error': error_7point,
        'F_8point': F_8point,
        'F_7point': F_7point
    }

results = compare_methods(points1, points2)
print("8-point error:", results['8-point error'])
print("7-point error:", results['7-point error'])
print("Fundamental matrix (8-point):", results['F_8point'])
print("Fundamental matrix (7-point):", results['F_7point'])

# In[12]:
def compute_fundamental_matrix(img1_path, img2_path):
    """
    Compute the fundamental matrix from two images using SIFT features and RANSAC.
    
    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        
    Returns:
        F (numpy.ndarray): 3x3 fundamental matrix.
        pts1 (numpy.ndarray): Nx2 array of points in the first image.
        pts2 (numpy.ndarray): Nx2 array of points in the second image.
    """
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Filter good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
    
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Compute fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # Select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    return F, pts1, pts2

# Example usage
img1_path = 'rendered_image_0.png'
img2_path = 'rendered_image_1.png'
F, pts1, pts2 = compute_fundamental_matrix(img1_path, img2_path)
print("Fundamental Matrix:\n", F)

# In[13]:
def recover_pose_from_fundamental(F, K, pts1, pts2):
    """
    Recover the rotation and translation from the fundamental matrix and intrinsic matrix.
    
    Args:
        F (numpy.ndarray): 3x3 fundamental matrix.
        K (numpy.ndarray): 3x3 intrinsic matrix.
        pts1 (numpy.ndarray): Nx2 array of points in the first image.
        pts2 (numpy.ndarray): Nx2 array of points in the second image.
        
    Returns:
        R (numpy.ndarray): 3x3 rotation matrix.
        t (numpy.ndarray): 3x1 translation vector.
    """
    # Compute the essential matrix
    E = K.T @ F @ K
    
    # Recover the pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t

def camera_pose_from_fundamental(F, K):
    """
    Compute four possible camera poses from fundamental matrix
    Args:
        F: 3x3 fundamental matrix
        K: 3x3 camera intrinsic matrix
    Returns:
        list of four possible (R,t) pairs
    """
    # Convert F to Essential matrix
    E = K.T @ F @ K
    
    # Normalize Essential matrix
    U, S, Vh = np.linalg.svd(E)
    S = np.diag([1, 1, 0])  # Force singular values to [1,1,0]
    E = U @ S @ Vh
    
    # Ensure determinant is positive
    if np.linalg.det(U @ Vh) < 0:
        Vh = -Vh
    
    # Create W matrix for decomposition
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Extract rotation matrices
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh
    
    # Extract translation vector
    t = U[:, 2]
    
    # Ensure proper rotation matrices (det = 1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        
    # Generate four possible poses
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    return poses

R,t = recover_pose_from_fundamental(F, params.intrinsic.intrinsic_matrix, points1, points2)
R2, t2 = recover_pose_from_fundamental(results['F_8point'], params.intrinsic.intrinsic_matrix, points1, points2)
R3, t3 = recover_pose_from_fundamental(results['F_7point'], params.intrinsic.intrinsic_matrix, points1, points2)

poses_8point = camera_pose_from_fundamental(results['F_8point'], params.intrinsic.intrinsic_matrix)
poses_7point = camera_pose_from_fundamental(results['F_7point'], params.intrinsic.intrinsic_matrix)

R,t = recover_pose_from_fundamental(F, params.intrinsic.intrinsic_matrix, points1, points2)
print(rotation_matrix_to_axis_angle(R))
for (R,t) in poses_7point:
    print(rotation_matrix_to_axis_angle(R))
for (R,t) in poses_8point:
    print(rotation_matrix_to_axis_angle(R))


between_transform = np.identity(4)
between_transform[:3,:3] = R
between_transform[:3,3] = t.flatten()   

# In[14]:
# new_rot = (extrinsics_2 @ between_transform)[:3, :3]
new_rot = extrinsics_2[:3,:3] @ R.T
# new_rot = R.T @ extrinsics_2[:3,:3]
new_trans = extrinsics_2[:3, 3]

params = o3d.camera.PinholeCameraParameters()
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
camera_transform = np.identity(4)
camera_transform[:3, :3] = new_rot
camera_transform[:3, 3] = new_trans
vis.add_geometry(car_mesh)
ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(params)
ctr.set_lookat(camera_transform[:3, 3])
ctr.set_front(camera_transform[:3, 2])
ctr.set_up(camera_transform[:3, 1])
print(params.extrinsic)
print(params.intrinsic.intrinsic_matrix)
vis.poll_events()
vis.update_renderer()

image = vis.capture_screen_float_buffer()
time.sleep(1)
vis.remove_geometry(car_mesh)
# vis.destroy_window()

image_path = f"rendered_image_3.png"
o3d.io.write_image(image_path, o3d.geometry.Image((np.asarray(image) * 255).astype(np.uint8)))
vis.destroy_window()

# %%
