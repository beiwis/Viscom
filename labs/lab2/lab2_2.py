import matplotlib
matplotlib.use('TkAgg')
from plotData_v2 import drawRefSystem, plotNumberedImagePoints
from cv2 import cvtColor, imread, COLOR_BGR2RGB
from lab2_1 import SVD_triangulation
import matplotlib.pyplot as plt
import numpy as np

def draw_epipolar_line(F: np.array, point: list[tuple], color: str = 'g'):
    """
    Plots the epipolar lines on image 2 given the fundamental matrix between two images and the point(s) on image 1.

    Args:
        F (numpy.matrix): The fundamental matrix between two images.
        x1 (numpy.matrix): The point(s) on image 1.
    """
    # Load the image
    img2 = cvtColor(imread('data/image2.png'), COLOR_BGR2RGB)

    # Compute the epipolar lines
    print(f'Point: \n{point}\nAppended point: \n{np.append(point, 1)}\n\nF: \n{F}')
    l2 = F @ np.append(point, 1)

    # Define the line
    l = [(-l2[2] - l2[0] * 0) / l2[1], (-l2[2] - l2[0] * img2.shape[1] ) / l2[1]]

    plt.plot([0, img2.shape[1]], l, c=color)
    plt.title('Image 2')
    plt.show()
    plt.draw()

def get_essential_matrix(T0: np.array, T1: np.array) -> np.array:
    """
    Compute the essential matrix from two World to Camera poses.

    Args:
    T0, T1 (np.array): The camera poses from both cameras (c1, c2) in world reference.

    Returns:
    The essential matrix.
    """
    # Extract the rotation and translation components
    R1, t1 = T0[:3, :3], T0[:3, 3]
    R2, t2 = T1[:3, :3], T1[:3, 3]

    # Compute the relative rotation and translation (slide 2.16: R = R_c1c0; t = t_c1c0)
    R = R2 @ R1.T
    t = t2 - R @ t1

    # Compute the skew-symmetric matrix of t (slide 2.16)
    t = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

    # Compute the essential matrix (slide 2.16: 𝑬 = [𝒕] × 𝑹)
    E = t @ R

    return E

def get_fundamental_matrix(E: np.array, K1: np.array, K2: np.array) -> np.array:
    """
    Compute the fundamental matrix from the intrinsic camera parameters and the essential matrix.

    Args:
    -----------
    E (np.array): The essential matrix.
    K1, K2 (np.array): The intrinsic camera parameters matrix from both cameras (c1, c2).

    Returns:
    --------
    np.array: The normalized fundamental matrix.
    """
    # Compute the fundamental matrix (slide 2.20: F = K)
    F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
    # Normalize the fundamental matrix
    F = F / F[-1, -1]
    return F

def get_fundamental_matrix(x1:np.array, x2:np.array, normalize:bool = False):
    # Normalize the points
    
    # Construct the A matrix
    # Solve for F (SVD)
    # De-normalize the matrix    
    pass

def get_camera_solutions(R1: np.array, R2: np.array, T1: np.array, T2: np.array, K1: np.array, K2: np.array, x1: np.array, x2: np.array):
    """
    Get the camera matrices and then triangulate the points.

    Args:
    R1, R2 (np.array): The rotation matrices.
    T1, T2 (np.array): The translation vectors.
    K1, K2 (np.array): The intrinsic camera parameters matrix from both cameras (c1, c2)
    x1, x2 (np.array): The matches from both cameras (c1, c2) in pixel coordinates.

    Returns:
    The four possible camera matrices
    """
    # Define the identity matrix and zero vector
    I = np.eye(3)
    zero = np.zeros((3, 1))

    # Define the four possible camera matrices
    P1 = K1 @ np.hstack((I, zero))
    P2_a = K2 @ np.hstack((R1, T1))
    P2_b = K2 @ np.hstack((R1, T2))
    P2_c = K2 @ np.hstack((R2, T1))
    P2_d = K2 @ np.hstack((R2, T2))

    # Triangulate the points
    X_a = SVD_triangulation(x1, x2, P1, P2_a)
    X_b = SVD_triangulation(x1, x2, P1, P2_b)
    X_c = SVD_triangulation(x1, x2, P1, P2_c)
    X_d = SVD_triangulation(x1, x2, P1, P2_d)

def get_camera_motion(E: np.array):
    """
    Compute the camera motion from the essential matrix.

    Args:
    E (np.array): The essential matrix.

    Returns:
    The rotation matrix R and the translation vector T.
    3D points in the world coordinate system.
    """
    F = get_fundamental_matrix(x1, x2)
    # Compute the essential matrix
    E = get_essential_matrix
    # Compute the possible camera motions
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    T1 = U[:, 2].reshape(-1, 1)
    T2 = -U[:, 2].reshape(-1, 1)
    return R1, R2, T1, T2

def Epipolar_lines_visualization(img1: np.array, img2: np.array, F: np.array):
    """
    Visualize the epipolar lines on the second image given the fundamental matrix between two images, selecting five points in the first image.
    This function corresponds to the first part of the second exercise of the second lab.
    Laboratory Session 2: Homography, Fundamental Matrix and Two View SfM
        2. Fundamental matrix and Structure from Motion
            2.1 Epipolar lines visualization
    Parameters:
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.
    F : numpy.ndarray
        The fundamental matrix between the two images.
    """
    # Display the images in a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the first image
    ax1.imshow(img1)
    ax1.set_title('Select points')
    plt.draw()

    # Colors for the epipolar lines
    colors = ['r', 'y', 'g', 'c', 'm']


    points = []
    for i in range(1):
        # Capture the point
        point = plt.ginput(1)  # Capture a single point
        points.append(point[0])
        ax1.plot(point[0][0], point[0][1], 'x', c=colors[i])  # Plot the selected point

        # Display the second image
        ax2.set_title('Epipolar line(s)')
        ax2.imshow(img2)
        ax2.axis('equal')

        # Draw the epipolar lines
        for i, point in enumerate(points):
            draw_epipolar_line(F, point, colors[i])

        plt.draw()

def Camera_poses_to_fundamental_matrix(T1, T2, E, K, img1, img2):
    E = get_essential_matrix(T1, T2)
    F = get_fundamental_matrix(E, K, K)
    np.set_printoptions(precision=7, suppress=True)
    print(f'Fundamental matrix: \n{F}')
    # Load the testing fundamental matrix
    F_test = np.loadtxt('data/F_21_test.txt')
    print(f'Testing fundamental matrix: \n{F_test}')
    Epipolar_lines_visualization(img1, img2, F)

                                       
if __name__ == '__main__':
    # 2.0 LOAD THE DATA
    #######################################
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

    # 2.1 EPIPOLAR LINES VISUALIZATION
    #######################################
    # Epipolar_lines_visualization(img1, img2, F)

    # 2.2 FUNDAMENTAL MATRIX DEFINITION
    #######################################
    # Camera_poses_to_fundamental_matrix(T1, T2, E, K, img1, img2)
    

    # # Load the matches
    # x1 = np.loadtxt('x1Data.txt')
    # x2 = np.loadtxt('x2Data.txt')

    # # Get the fundamental matrix
    # F = get_fundamental_matrix(x1, x2)
    # # Normalize the fundamental matrix
    # F = F / F[-1, -1]
    # print(f"Fundamental matrix: {F}")
    # # Get the testing fundamental matrix
    # F_test = np.loadtxt('F_21_test.txt')
    # print(f"Fundamental matrix test: {F_test}")
##############################################################################
#                         Plot the point on the images                       #
##############################################################################
# # Load the images
# img1 = cvtColor(imread('image1.png'), COLOR_BGR2RGB)
# img2 = cvtColor(imread('image2.png'), COLOR_BGR2RGB)
# # Load the points
# x1 = np.loadtxt('x1Data.txt')
# x2 = np.loadtxt('x2Data.txt')
# # Plot the images and the points
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
# plt.scatter(x1[0, :], x1[1, :], marker='x', c='r')
# plt.title('Image 1')
# plt.subplot(122)
# plt.imshow(img2)
# plt.scatter(x2[0, :], x2[1, :], marker='x', c='r')
# plt.title('Image 2')
# plt.draw()  # We update the figure display
# print('Click in the image to continue...')
# plt.waitforbuttonpress()
