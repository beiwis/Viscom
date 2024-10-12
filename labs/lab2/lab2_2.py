import matplotlib
matplotlib.use('TkAgg')
from plotData_v2 import drawRefSystem, plotNumberedImagePoints
from cv2 import cvtColor, imread, COLOR_BGR2RGB
from lab2_1 import SVD_triangulation
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    # Load the images
    img1 = cvtColor(imread('data/image1.png'), COLOR_BGR2RGB)
    img2 = cvtColor(imread('data/image2.png'), COLOR_BGR2RGB)
    # Load the intrinsic camera parameters
    K = np.loadtxt('data/K_c.txt')
    # Load the camera poses (world reference)
    T1 = np.loadtxt('data/T_w_c1.txt')
    T2 = np.loadtxt('data/T_w_c2.txt')
    # Load the matches coordinates in pixel coordinates for camera 1 and 2
    x1 = np.loadtxt('data/x1Data.txt')
    x2 = np.loadtxt('data/x2Data.txt')
    # Load the F matrix (theoretical, only for testing)
    F = np.loadtxt('data/F_21_test.txt')
    return img1, img2, K, T1, T2, x1, x2, F

def draw_epipolar_line(F: np.array, point: tuple, img2: np.array, color: str = 'g'):
    """
    Plots the epipolar lines on image 2 given the fundamental matrix between two images and the point(s) on image 1.

    Args:
        F (numpy.matrix): The fundamental matrix between two images.
        point (numpy.matrix): The point on image 1.
        img2 (np.array): The second image for the Epipolar lines to be plotted on
        color (str): The color of the line
    """
    # Compute the epipolar lines
    print(f'Point: \n{point}')
    l2 = F @ np.append(point, 1)

    # Define the line
    l = [(-l2[2] - l2[0] * 0) / l2[1], (-l2[2] - l2[0] * img2.shape[1] ) / l2[1]]

    plt.plot([0, img2.shape[1]], l, c=color)
    plt.title('Epipolar lines on Image 2')
    plt.draw()

def get_fundamental_matrix(*args) -> np.array:
    """
    Compute the fundamental matrix.

    This function can compute the fundamental matrix in two ways:
    1. From the intrinsic camera parameters and the essential matrix.
    2. Using the eight-point algorithm from matched pixel coordinates.

    Args:
        -  Case 1: 
            T1 (np.array): The camera pose from camera 1 (c1) in world reference.
            T2 (np.array): The camera pose from camera 2 (c2) in world reference.
            K1 (np.array): The intrinsic camera parameters matrix from camera 1 (c1).
            K2 (np.array): The intrinsic camera parameters matrix from camera 2 (c2).
        
        -  Case 2:
            x1 (np.array): The points from the first camera (c1) in pixel coordinates.
            x2 (np.array): The points from the second camera (c2) in pixel coordinates.

    Returns:
        np.array: The 3x3 fundamental matrix.
    """
    # Case 1: Get the fundamental matrix from the intrinsic camera parameters and the essential matrix.
    if len(args) == 4 and all(isinstance(arg, np.ndarray) for arg in args):
        # Case 1: Using camera parameters and poses
        T1, T2, K1, K2 = args
        
        # Compute the relative rotation and translation (T_2_1 = T2^-1 @ T1)
        T_2_1 = np.linalg.inv(T2) @ T1

        # Extract the rotation and translation components
        R, t = T_2_1[:3, :3], T_2_1[:3, 3]

        # Compute the skew-symmetric matrix of t
        t = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])

        # Compute the essential matrix
        E = t @ R

        # Compute the fundamental matrix
        F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)

        return F
    # Case 2: Get the fundamental matrix using the eight-point algorithm from matched pixel coordinates.
    elif len(args) == 2 and all(isinstance(arg, np.ndarray) for arg in args):
        # Case 2: Using the eight-point algorithm
        x1, x2 = args

        # Construct the A matrix
        # We get 8 out of all the possible matches:
        indices = np.random.choice(x1.shape[1], 8, replace=False)
        A = []
        for i in indices:
            A.append([
                x2[0, i] * x1[0, i], x2[0, i] * x1[1, i], x2[0, i],
                x2[1, i] * x1[0, i], x2[1, i] * x1[1, i], x2[1, i],
                x1[0, i], x1[1, i], 1
            ])
        A = np.array(A)

        # Solve for F (SVD)
        _, _, Vh = np.linalg.svd(A)
        F = Vh[-1, :].reshape(3, 3)

        return F

    else:
        raise ValueError("Invalid arguments. Provide either (T1, T2, K1, K2) or (x1, x2).")

def Epipolar_lines_visualization(img1: np.array, img2: np.array, F: np.array, n: int=1):
    """
    Visualize the epipolar lines on the second image given the fundamental matrix between two images, selecting five points in the first image.
    This function corresponds to the first part of the second exercise of the second lab.\n
    Laboratory Session 2: Homography, Fundamental Matrix and Two View SfM\n
        2. Fundamental matrix and Structure from Motion\n
            2.1 Epipolar lines visualization

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        F (numpy.ndarray): The fundamental matrix between the two images.
        n (int): The number of points/epipolar lines.
    """    
    # Display the images in a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the first image
    ax1.imshow(img1)
    ax1.set_title('Select points')
    plt.draw()

    # Colors for the epipolar lines
    colors = ['r', 'y', 'g', 'c', 'm']
    for i in range(n):
        # Capture the point
        point = plt.ginput(1)  # Capture a single point
        ax1.plot(point[0][0], point[0][1], 'x', c=colors[i%5])  # Plot the selected point

        # Display the second image
        ax2.set_title('Epipolar line(s)')
        ax2.imshow(img2)
        ax2.axis('equal')

        # Draw the epipolar lines
        draw_epipolar_line(F, point, img2, colors[i%5])

        print(f"Degub: pedopedo n:{i}")

    plt.show()

def Camera_poses_to_fundamental_matrix(T1, T2, K, img1, img2):
    # Load the testing fundamental matrix
    F_test = np.loadtxt('data/F_21_test.txt')
    print(f'Testing fundamental matrix: \n{F_test}')
    Epipolar_lines_visualization(img1, img2, F_test)
    # Calculate the F matrix using the different camera poses
    F = get_fundamental_matrix(T1, T2, K, K)
    print(f'Fundamental matrix: \n{F}')
    Epipolar_lines_visualization(img1, img2, F)

def Fundamental_matrix_8_point_solution(x1, x2, img1, img2, F):
    F = get_fundamental_matrix(x1, x2)
    Epipolar_lines_visualization(img1, img2, F, 5)

if __name__ == '__main__':
    # 2.0 LOAD THE DATA
    #######################################
    img1, img2, K, T1, T2, x1, x2, F = load_data()

    # 2.1 EPIPOLAR LINES VISUALIZATION
    #######################################
    # Epipolar_lines_visualization(img1, img2, F)

    # 2.2 FUNDAMENTAL MATRIX DEFINITION
    #######################################
    # Camera_poses_to_fundamental_matrix(T1, T2, K, img1, img2)
    
    # 2.3 FUNDAMENTAL MATRIX LINEAR ESTIMATION WITH EIGHT POINT SOLUTION
    ######################################################################### FIXME: i get the epipole in the middle of the screen?
    # Fundamental_matrix_8_point_solution(x1, x2, img1, img2, F)

    # TODO: 2.4 POSE ESTIMATION FROM TWO VIEWS 
    ########################################
    def estimate_pose_from_essential_matrix(E, K):
        """
        Estimate the four possible camera poses from the essential matrix.

        Args:
            E (np.array): The essential matrix.
            K (np.array): The intrinsic camera parameters matrix.

        Returns:
            list: A list of four possible camera poses (3x4 matrices).
        """
        # Perform SVD on the essential matrix
        U, _, Vt = np.linalg.svd(E)

        # Ensure a proper rotation matrix
        if np.linalg.det(U @ Vt) < 0:
            Vt = -Vt

        # Define the W matrix
        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        # Compute the four possible camera poses
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2]

        poses = [
            np.hstack((R1, t.reshape(-1, 1))),
            np.hstack((R1, -t.reshape(-1, 1))),
            np.hstack((R2, t.reshape(-1, 1))),
            np.hstack((R2, -t.reshape(-1, 1)))
        ]

        return poses

    def triangulate_points(P1, P2, x1, x2):
        """
        Triangulate 3D points from two views.

        Args:
            P1 (np.array): The projection matrix for the first camera.
            P2 (np.array): The projection matrix for the second camera.
            x1 (np.array): The points from the first camera (c1) in pixel coordinates.
            x2 (np.array): The points from the second camera (c2) in pixel coordinates.

        Returns:
            np.array: The triangulated 3D points.
        """
        points_3d = []
        for i in range(x1.shape[1]):
            A = np.array([
                x1[0, i] * P1[2, :] - P1[0, :],
                x1[1, i] * P1[2, :] - P1[1, :],
                x2[0, i] * P2[2, :] - P2[0, :],
                x2[1, i] * P2[2, :] - P2[1, :]
            ])
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            points_3d.append(X / X[3])

        return np.array(points_3d).T

    def find_correct_pose(poses, K, x1, x2):
        """
        Find the correct camera pose from the four possible poses.

        Args:
            poses (list): A list of four possible camera poses (3x4 matrices).
            K (np.array): The intrinsic camera parameters matrix.
            x1 (np.array): The points from the first camera (c1) in pixel coordinates.
            x2 (np.array): The points from the second camera (c2) in pixel coordinates.

        Returns:
            np.array: The correct camera pose (3x4 matrix).
        """
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        for pose in poses:
            P2 = K @ pose
            points_3d = triangulate_points(P1, P2, x1, x2)
            if np.all(points_3d[2, :] > 0):  # Check if all points are in front of both cameras
                return pose
        raise ValueError("No valid pose found.")

    # Compute the essential matrix from the fundamental matrix
    E = K.T @ F @ K

    # Estimate the four possible camera poses
    poses = estimate_pose_from_essential_matrix(E, K)

    # Find the correct camera pose
    correct_pose = find_correct_pose(poses, K, x1, x2)
    print(f'Correct camera pose: \n{correct_pose}')

