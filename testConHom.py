import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.transform import warp
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
import math
from matplotlib.patches import Ellipse

# Gets the vector spanning the rank-1 matrix Hs
def get_vech(H):
    h = [0, 0, 0]
    h[0] = math.sqrt(H[0,0])
    h[1] = math.sqrt(H[1,1])
    h[2] = math.sqrt(H[2,2])
    h[1] = np.sign(H[0,1]) * h[1]
    h[2] = np.sign(H[0,2]) * h[2]

    return h

# Returns the homogenous matrix representation of an ellipse given its center (xc, yc), axes (a, b) and angle
def ell_to_hom(xc, yc, a, b, angle):
    A = b**2 * np.cos(angle)**2 + a**2 * np.cos(angle)**2
    B = a**2 * np.cos(angle) * np.sin(angle) - b**2 * np.cos(angle) * np.sin(angle)
    C = a**2 * np.cos(angle)**2 + b**2 * np.sin(angle)**2
    D = -b**2 * xc * np.cos(angle) - a**2 * yc * np.sin(angle)
    E = -a**2 * yc * np.cos(angle) + b**2 * xc * np.sin(angle)
    F = -a**2 * b**2 + b**2 * xc**2 + a**2 * yc**2

    return np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])/(-F)

# Returns the second degree polynomial of an ellipse given its homogeneous matrix representation
def hom_to_eq(M):
    A = M[0,0]
    B = M[0,1]*2
    C = M[1,1]
    D = M[0,2]*2
    E = M[1,2]*2
    F = M[2,2]

    return A, B, C, D, E, F

# Returns the second degree polynomial of an ellipse given its center (xc, yc), axes (a, b) and angle
def ell_to_eq(xc, yc, a, b, angle):
    A = b**2 * np.cos(angle)**2 + a**2 * np.cos(angle)**2
    B = a**2 * np.cos(angle) * np.sin(angle) - b**2 * np.cos(angle) * np.sin(angle)
    C = a**2 * np.cos(angle)**2 + b**2 * np.sin(angle)**2
    D = -b**2 * xc * np.cos(angle) - a**2 * yc * np.sin(angle)
    E = -a**2 * yc * np.cos(angle) + b**2 * xc * np.sin(angle)
    F = -a**2 * b**2 + b**2 * xc**2 + a**2 * yc**2

    return A, B, C, D, E, F
    
# Returns the parameters of an ellipse given its second degree polynomial
def eq_to_par(A, B, C, D, E, F, Q):

    xc = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    yc= (2 * A * E - B * D) / (B**2 - 4 * A * C)

    a = math.sqrt(2 * (- np.linalg.det(Q[:3,:3]) / np.linalg.det(Q[:2,:2])) / (A + C - math.sqrt(((A - C)**2 + B**2))))
    b = math.sqrt(2 * (- np.linalg.det(Q[:3,:3]) / np.linalg.det(Q[:2,:2])) / (A + C + math.sqrt(((A - C)**2 + B**2))))
    #a = 10* np.random.rand()
    #b = 10* np.random.rand()

    angle = math.atan2(C - A + math.sqrt(((A - C)**2 + B**2)), B)

    return xc, yc, a, b, angle

# Homography estimation given the matrix representations C[i] of the original conics and the matrix represenations Q[i] of their images under said homography
def get_H(C, Q_inv):

    a = [math.sqrt(1/M[0, 0]) for M in C]
    b = [math.sqrt(1/M[1, 1]) for M in C]

    #print(Q_inv)

    Hx1 = ((Q_inv[0] - Q_inv[1])*(b[0]**2-b[2]**2) - (Q_inv[0] - Q_inv[2])*(b[0]**2 - b[1]**2))/((a[0]**2 -a[1]**2)*(b[0]**2 -b[2]**2) - (a[0]**2 - a[2]**2)*(b[0]**2 -b[1]**2))
    
    Hx2 = (Q_inv[0] - Q_inv[2] - (a[0]**2 - a[2]**2) * Hx1) / (b[0]**2 - b[2]**2)

    Hx3 = -Q_inv[0] + a[0]**2 * Hx1 + b[0]**2 * Hx2


    #print("Hx1 = ", Hx1)
    h1 = get_vech(Hx1)
    #print(h1)

    #print("Hx2 = ", Hx2)
    h2 = get_vech(Hx2)
    #print(h2)

    #print("Hx3 = ", Hx3)
    h3 = get_vech(Hx3)
    #print(h3)

    H = np.transpose(np.vstack((h1, h2, h3)))
    # print("heyhey H = ", H)
    return H/H[2,2]

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
        return randHom()

# This function gives a random intrinsic camera matrix
def randIntrinsic():
    K =  np.array([[np.random.rand(), np.random.rand(), np.random.rand()],
        [0, np.random.rand(), np.random.rand()],
        [0, 0, np.random.rand()]])
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

def d2_plot_con(Q):
    
    plt.figure()
    for M in Q:
        x = -np.linspace(-20,20,1000)
        y = np.linspace(-20,20,1000)
    
        x,y = np.meshgrid(x,y)
       
        A, B, C, D, E, F = hom_to_eq(M)
    
        Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F 
        print(Z)
        plt.contour(x, y,(Z),[0])
    plt.title("Original")
    plt.show()
    # plt.xlim([-1.5,1.5])
    # plt.ylim([-11.5,-8.5])


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

def big_plot_con(original, image, back):
    fig, axis = plt.subplots(1, 3)
    
    ells = []
    for M in original:
        A, B, C, D, E, F = hom_to_eq(M)
        xc, yc, a, b, an = eq_to_par(A, B, C, D, E, F, M)
        ells.append(Ellipse(xy=(xc, yc), width=a, height=b, angle=an/math.pi*180, linewidth=2, fill=False, zorder=2))

    for e in ells:
        axis[0].add_artist(e)
        # e.set_clip_box(axis[0].bbox)
        # e.set_alpha(np.random.rand())
        # e.set_facecolor(np.random.rand(3))

    ells = []
    for M in image:
        A, B, C, D, E, F = hom_to_eq(M)
        xc, yc, a, b, an = eq_to_par(A, B, C, D, E, F, M)
        ells.append(Ellipse(xy=(xc, yc), width=a, height=b, angle=an/math.pi*180, linewidth=2, fill=False, zorder=2))

    for e in ells:
        axis[1].add_artist(e)
        # e.set_clip_box(axis[1].bbox)
        # e.set_alpha(1)
        # e.set_facecolor((0,0,0))

    ells = []
    for M in back:
        A, B, C, D, E, F = hom_to_eq(M)
        xc, yc, a, b, an = eq_to_par(A, B, C, D, E, F, M)
        ells.append(Ellipse(xy=(xc, yc), width=a, height=b, angle=an/math.pi*180, linewidth=2, fill=False, zorder=2))


    for e in ells:
        axis[2].add_artist(e)
        # e.set_clip_box(axis[2].bbox)
        # e.set_alpha(np.random.rand())
        # e.set_facecolor(np.random.rand(3))

    
    axis[0].set_xlim(-5, 5)
    axis[0].set_ylim(-5, 5)
    axis[1].set_xlim(-5, 5)
    axis[1].set_ylim(-5, 5)
    axis[2].set_xlim(-5, 5)
    axis[2].set_ylim(-5, 5)
    axis[0].set_box_aspect(1)
    axis[1].set_box_aspect(1)
    axis[2].set_box_aspect(1)
    axis[0].set_title("original")
    axis[1].set_title("image")
    axis[2].set_title("back")

    plt.show()
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

def test_con(s1, s2):
    a = [1, 2, 3]
    b = [6, 4, 2]

    C = [np.array([[1/a[i]**2, 0, 0], [0, 1/b[i]**2, 0], [0, 0, -1]]) for i in range(3)]
    preH = randHom(3)
    preH =  np.array(preH)
    
    # print("preH =", preH)


    Q_inv = [preH @ np.linalg.inv(M) @ np.transpose(preH) for M in C]
    Q = [np.linalg.inv(M) for M in Q_inv]
   
    d2_plot_con(C)
    d2_plot_con(Q)
    d2_plot_con([np.transpose(preH) @ M @ preH for M in Q])
    # print("THe image of C[0] =", Q[0])

    # A1, B1, C1, D1, E1, F1 = hom_to_eq(Q[0])

    # z1, z2, z3, z4, z5 = eq_to_par(A1, B1, C1, D1, E1, F1, Q[0])
    # print("I got: ", [np.transpose(preH) @ M @ preH for M in Q])

    # print(C[1])
    # print(Q)
    # A1, B1, C1, D1, E1, F1 = hom_to_eq(C[1])
    # print(A1, B1, C1, D1, E1, F1)
    # z1, z2, z3, z4, z5 = eq_to_par(A1, B1, C1, D1, E1, F1, C[1])
    # print(ell_to_eq(z1, z2, z3, z4, z5))
    H = get_H(C, Q_inv)
    # print("preH = ", preH)
    # print("H = ", H)

   # print("C should be =", [np.transpose(preH) @ M @ preH for M in Q])
    # print("Q = ", Q[0])
    # Option to print the conics, their image, and the backprojection under our estimated H
    #big_plot_con(C, C, C)

    return np.linalg.norm(H - preH)


print("Fnorm = ", test_con(0,0))
# test_plot(50, 0, 0.001, 0.0001)