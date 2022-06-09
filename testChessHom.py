import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.transform import warp
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
lenz = 1
import math

# Makes an Euclidean vector homogenous
def e2h(x):
    return np.hstack((x, [1]))

# This function creates a random 3d rotation matrix
def randRot(n):
    M = np.random.rand(n, n)
    U, _, V = np.linalg.svd(M, full_matrices=False)
    R = np.linalg.det(U) * U #to make sure R has determinant 1
    return R

def randHom(n):
    H = np.random.rand(n, n)
    if np.linalg.det(H) != 0:
        return H/H[2,2]
    else:
        return randHom(n)

# This function gives a random intrinsic camera matrix
def randIntrinsic():
    K =  np.array([[np.random.rand(), np.random.rand(), np.random.rand()],
        [0, np.random.rand(), np.random.rand()],
        [0, 0, np.random.rand()]])
    # print(K[2, 2])
    return np.true_divide(K, K[2,2])

# This function is just for shorthand notation
def func_v(h, i, j):
    return np.array([h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j]])

# This function estimates the intrinsic camera using n given homographies (see Zhang)
def get_intrinsic(homographies, n):
    A = []
    
    for k, h in enumerate(homographies):
        row_1 = func_v(h, 0, 1)
        row_2 = func_v(h, 0, 0) - func_v(h, 1, 1)
        if k  != 0:
            A =np.vstack((A, row_1, row_2))
        else:
            A = np.vstack((row_1, row_2))
        
    _, _, V = np.linalg.svd(A)
    b = V[-1]

    B = np.array([[b[0], b[1], b[3]],
                     [b[1], b[2], b[4]],
                     [b[3], b[4], b[5]]])

    v0 = (B[1, 0] * B[2, 0] - B[0, 0] * B[2, 1])/(B[0, 0] * B[1, 1] - B[1, 0]* B[1, 0])
    lamb = B[2, 2] - (B[2, 0] * B[2, 0] + v0 * (B[1, 0] * B[2, 0] - B[0, 0] * B[2, 1]))/B[0, 0]
    alpha =  np.sqrt(lamb / B[0, 0])
    beta = np.sqrt(lamb * B[0, 0] / (B[0, 0] * B[1, 1] - B[1, 0] * B[1, 0]))
    gamma = -B[1, 0] * alpha**2 * beta / lamb
    u0 = gamma * v0 / beta - B[2, 0] * alpha**2 / lamb

    K =  np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return K

# DLT homography estimation given point correspondencs H * objpoints[i] = imgpoints[i]
def DLT(objpoints, imgpoints):
    A = []
    
    for k in range(len(objpoints)):
        row_1 = np.hstack((np.array([0,0,0]), -imgpoints[k][2] * objpoints[k], imgpoints[k][1] * objpoints[k]))
        row_2 = np.hstack((imgpoints[k][2] * objpoints[k], np.array([0,0,0]), -imgpoints[k][0] * objpoints[k]))
        if k  != 0:
            A =np.vstack((A, row_1, row_2))
        else:
            A = np.vstack((row_1, row_2))
        
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3,3)
    return P/P[2,2]

#
#
# This concludes the backcore, below are testing-purpose plot functions
#
#

def d3_plot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    for p in points:
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])


    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    time.sleep(lenz)
    plt.close('all')


def d2_plot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = []
    y = []
    for p in points:
        x.append(p[0]/p[2])
        y.append(p[1]/p[2])

    ax.scatter(x, y, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.show()
    time.sleep(lenz)
    plt.close('all')


def big_plot(objpoints, imgpoints, backpoints):
    fig, axis = plt.subplots(1, 3)


    xp = []
    yp = []
    for p in objpoints:
        xp.append(p[0]/p[2])
        yp.append(p[1]/p[2])

    axis[0].scatter(xp, yp, c='r', marker='o')

    xi = []
    yi = []
    for p in imgpoints:
        xi.append(p[0]/p[2])
        yi.append(p[1]/p[2])

    axis[1].scatter(xi, yi, c='r', marker='o')

    xb = []
    yb = []
    for p in backpoints:
        xb.append(p[0]/p[2])
        yb.append(p[1]/p[2])

    axis[2].scatter(xb, yb, c='r', marker='o')

    axis[0].set_title("original")
    axis[1].set_title("image")
    axis[2].set_title("back")
    plt.tight_layout()
    plt.show()
    time.sleep(lenz)
    plt.close('all')

# This function runs tests for final plot
def test_plot(M, min_s, max_s, step_size):
    errors = []
    for s in np.arange(min_s, max_s, step_size):
        print("s = ", s)
        error = 0
        i = 0
        while i < M:
            i += 1
            print(i)
            res = test_DLT(0, s)
            # Test if calculation went well
            if(math.isnan(res)):
                i -= 1
            else:
                error += res
        errors.append(error/M)
    
    print(errors)
    plt.xlabel("data error variance")
    plt.ylabel("Error of estimation (Frobenius)")
    plt.title("Test camera callibration using DLT on erroneous data (chessboard)")
    plt.plot(np.arange(min_s, max_s, step_size), errors, 'ro')
    plt.show()

def get_vech(H):
    h = [0, 0, 0]
    h[0] = math.sqrt(H[0,0])
    h[1] = math.sqrt(H[1,1])
    h[2] = math.sqrt(H[2,2])
    h[1] = np.sign(H[0,1]) * h[1]
    h[2] = np.sign(H[0,2]) * h[2]

# Applies DLT (using a 7x7 chess pattern) on one random homography and tests the correctness of its estimate using the Frobenius norm
# The data of the image after the gets a random error added to it
def test_DLT(s1, s2):

    # Create a random homography
    preH = randHom(3)

    # Create a chess pattern (with homogeneous coordinates)
    objpoints = np.zeros((7*7, 3), np.float32)
    objpoints[:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)  
    objpoints[:,2:] =1  

    # Create the image of our chess board
    imgpoints = [preH @ X for X in objpoints]
    #imgpoints = objpoints

    # Apply Gaussian error
    objpoints = [np.array(X) + np.random.multivariate_normal([0, 0, 0], [[s1, 0, 0], [0, s1, 0], [0, 0, s1]])  for X in objpoints]

    # Apply Gaussian error
    imgpoints = [np.array(X) + np.random.multivariate_normal([0, 0, 0], [[s2, 0, 0], [0, s2, 0], [0, 0, s2]])  for X in imgpoints]
 
    # Estimate the homography H using Direct Linear Transform
    H = DLT(objpoints, imgpoints)
    
    # Option to print the chessboard, its image, and the backprojection under our estimated H
    # big_plot(objpoints, imgpoints, [np.linalg.inv(H) @ X for X in imgpoints])


    # Return the Frobenius distance between our estimate and the real homography
    return np.linalg.norm(preH -  H)

# print(test_DLT(0,0.001, ))
test_plot(500, 0, 0.001, 0.00001)

