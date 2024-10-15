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
    # Load the ground truth 3D points
    X_w = np.loadtxt('data/X_w.txt')
    # Load the the plane equation coefficients in camera 1 reference
    Pi_1 = np.loadtxt('data/Pi_1.txt')
    # Load the floor matches coordinates in pixel coordinates for camera 1 and 2
    x1_floor = np.loadtxt('data/x1FloorData.txt')
    x2_floor = np.loadtxt('data/x2FloorData.txt')

    return img1, img2, K, T1, T2, x1, x2, F, X_w, Pi_1, x1_floor, x2_floor

def get_homography(T_w_1, T_w_2, K1, K2, Pi_1):
    # Obtain T_2_1
    T_2_1 = np.linalg.inv(T_w_2) @ T_w_1
    # Extract rotation (R) and translation (t) from the poses
    R, t = T_2_1[:3, :3], T_2_1[:3, 3]
    # Extract the plane normal and distance to the origin
    n, d = Pi_1[:3], Pi_1[3]
    # Compute the homography 𝑯_21 = 𝑲_2[ 𝑹_𝑐2𝑐1 − 𝒕_𝑐2𝑐1 𝒏^𝑡 1/𝑑 ]𝑲_1^−1
    H = K2 @ (R - t @ n.T / d) @ np.linalg.inv(K1)
    return H
if __name__ == '__main__':
    
    # 3.0 Load the data
    ########################
    img1, img2, K, T_w_1, T_w_2, x1, x2, F, X_w, Pi_1, x1_floor, x2_floor = load_data()
    
    # 3.1 Homography definition
    ################################
    # Compute the homography matrix
    H = get_homography(T_w_1, T_w_2, K, K, Pi_1)
    print("Homography matrix H:\n", H)

    # 3.2 Point transfer visualization
    #######################################
    # Transfer points using the homography
    x1_floor_estimated = H @ x2_floor
    # Normalize the points
    for i in range(x1_floor_estimated.shape[1]):
        x1_floor_estimated[:, i] = x1_floor_estimated[:, i] / x1_floor_estimated[-1, i]
    print("x1_floor_estimated:\n", x1_floor_estimated)
    print("x1_floor_gt:\n", x1_floor)
    # for i in range(x2_floor.shape[1]):
    #     print("x2_floor_homogeneous:\n", x2_floor[:,i])
    #     x1_i = x2_floor[:, i] @ H
    #     x1_i = x1_i / x1_i[-1]
    #     print("x1_floor_estimated:\n", x1_i)
    #     print("x1_floor_gt:\n", x1_floor[:,i])
    #     print("\n###############################################\n")
    #     x2_floor_estimated = np.hstack((x1_floor_estimated, x1_i))
   

    # Plot the original and transferred points
    plt.figure(figsize=(10, 5))

    # Plot original points in image 1
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plotNumberedImagePoints(x1_floor, 'g')
    plotNumberedImagePoints(x1_floor_estimated.T, 'r')
    plt.title('Original points in Image 1')

    # Plot transferred points in image 2
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plotNumberedImagePoints(x2_floor, 'g')
    plt.legend()
    plt.title('Transferred points in Image 2')

    plt.show()