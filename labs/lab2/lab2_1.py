from plotData_v2 import drawRefSystem, plotNumberedImagePoints
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cvtColor, imread, COLOR_BGR2RGB
# np.set_printoptions(precision=4,linewidth=1024,suppress=True)

def compute_p(k_matrix: np.matrix, t_matrix: np.matrix) -> np.matrix:
    """
    Compute the projection matrix.

    Args:
    k_matrix (np.matrix): The intrinsic camera parameters matrix.
    t_matrix (np.matrix): The transformation matrix World to Camera.

    Returns:
    np.matrix: The projection matrix, World to Point.
    """
    # We get the inverse of the transformation matrix, the transformation Camera to World
    t_matrix_inverted = np.linalg.inv(t_matrix)
    canon_matrix = np.matrix([[1, 0, 0, 0], # TODO: why the canon matrix¿?
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    # We compute the projection matrix
    p_matrix = k_matrix @ canon_matrix @ t_matrix_inverted
    return p_matrix

def SVD_triangulation(x1: np.matrix, x2: np.matrix, P1: np.matrix, P2: np.matrix) -> np.matrix:
    """
    Triangulate the 3D position from two 2D points and their Projection matrices

    Args:
    x1, x2 (np.matrix): The points coodinates from both cameras (c1, c2)
    P1, P2 (np.matrix): The projection matrices from both cameras (c1, c2)

    Returns:
    The triangulated point, in 3D coordinates
    """
    # Create an empty 3D points matrix: 4 rows, 0 columns
    X_3d = np.empty((4, 0))
    for i in range(x1.shape[1]):   # for each point
        # Add a third value (1) in all the columns of x1 and x2
        x1_homogeneous = np.vstack((x1, np.ones((1, x1.shape[1]))))
        x2_homogeneous = np.vstack((x2, np.ones((1, x2.shape[1]))))
        # We compute the A matrix
        A = np.vstack((x1_homogeneous[0, i] * P1[2, :] - P1[0, :], x1_homogeneous[1, i] * P1[2, :] - P1[1, :], x2_homogeneous[0, i] * P2[2, :] - P2[0, :], x2_homogeneous[1, i] * P2[2, :] - P2[1, :]))
        # We compute the SVD
        _, _, V = np.linalg.svd(A)
        # We get the last column (row bc numpy) of V
        X = V[-1, :]
        # We normalize the 3D point
        X = X / X[-1]
        # We add the 3D point to the 3D points matrix
        X_3d = np.hstack((X_3d, X.reshape(-1, 1)))
    return X_3d

if __name__ == '__main__':
    ##############################################################################
    #        Calculate the projection matrices and triangulate the point         #
    ##############################################################################
    # Load the matrices
    T_w_c1 = np.loadtxt('data/T_w_c1.txt')
    T_w_c2 = np.loadtxt('data/T_w_c2.txt')
    K_c = np.loadtxt('data/K_c.txt')
    #Load the matches
    x1 = np.loadtxt('data/x1Data.txt')
    x2 = np.loadtxt('data/x2Data.txt')
    # Calculate the projection matrices
    P1 = compute_p(K_c, T_w_c1)
    P2 = compute_p(K_c, T_w_c2)
    # Triangulate the points
    X_w_SVD = SVD_triangulation(x1, x2, P1, P2)
    X_w = np.loadtxt('data/X_w.txt')
    # Print the points and errors
    print('3D point (Ground truth):')
    print(X_w[:,0])
    print('3D point (SVD):')
    print(X_w_SVD)

    ##############################################################################
    #                Plot the 3D cameras and the SVD triangulation               #
    ##############################################################################
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    # We plot the X_w_SVD
    ax.scatter(X_w_SVD[0, :], X_w_SVD[1, :], X_w_SVD[2, :], marker='.', label='SVD Triangulation')
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.', label='Ground Truth')
    #plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) 

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.title('SVD triangulation vs Ground Truth')
    plt.legend()
    plt.show()

    

