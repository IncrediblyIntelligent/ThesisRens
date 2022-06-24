import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.transform import warp
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
lenz = 1
import random
import math

def f(param,subX,subY):
    A,B,C,D,E,F = param
    som = 0
    for i in range(len(subX)):
        x = subX[i]
        y = subY[i]
        som += (A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F) ** 2 / ((2 * A * x + B * y + D) ** 2 + (B * x + 2 * C * y + E) ** 2)
    return som

def grad_f(param, subX, subY):
    A,B,c,d,E,F = param
    som = np.array([[0,0,0,0,0,0]])

    for i in range(len(subX)):
        x = subX[i]
        y = subY[i]
        som = som + np.array([[(-4*x*(d + 2*A*x + B*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*x**2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        -(((F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2*(2*y*(d + 2*A*x + B*y) + 2*x*(E + B*x + 2*c*y)))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2) + (2*x*y*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-4*y*(E + B*x + 2*c*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*y**2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-2*(d + 2*A*x + B*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*x*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-2*(E + B*x + 2*c*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*y*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)]])
    return som

def g(param, subX,subY):
    A,B,C,D,E,F = param
    i = random.randint(0, len(subX) - 1)
    x = subX[i]
    y = subY[i]
    return (A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F) ** 2 / ((2 * A * x + B * y + D) ** 2 + (B * x + 2 * C * y + E) ** 2)

def grad_g(param,subX,subY):
    A,B,c,d,E,F = param
    i = random.randint(0, len(subX) - 1)
    x = subX[i]
    y = subY[i]
    return np.array([[(-4*x*(d + 2*A*x + B*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*x**2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        -(((F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2*(2*y*(d + 2*A*x + B*y) + 2*x*(E + B*x + 2*c*y)))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2) + (2*x*y*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-4*y*(E + B*x + 2*c*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*y**2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-2*(d + 2*A*x + B*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*x*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (-2*(E + B*x + 2*c*y)*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2)**2)/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)**2 + (2*y*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2),
        (2*(F + d*x + A*x**2 + E*y + B*x*y + c*y**2))/((d + 2*A*x + B*y)**2 + (E + B*x + 2*c*y)**2)]])


# Gets the vector spanning the rank-1 matrix Hs
def get_vech(H):

    U, S, V = np.linalg.svd(H)

    return math.sqrt(S[0]) * np.transpose(U)[0]
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


    # print("Hx1 = ", Hx1)
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
    M = np.random.rand(n,n)
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

def d2_plot_PARM(A, B, C, D, E, F):
    
    plt.figure()
    x = -np.linspace(-7,7,1000)
    y = np.linspace(-7,7,1000)
    
    x,y = np.meshgrid(x,y)

    Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F 
    # print(Z)
    plt.contour(x, y,(Z),[0])
    plt.title("Image")
    plt.show()
    # plt.xlim([-1.5,1.5])
    # plt.ylim([-11.5,-8.5])

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

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('Before Rotation and Translation')
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
            print(i)
            i += 1
            res = test_cal(5, s)
            # Test if calculation went well
            if(math.isnan(res)):
                print("nan value encountered")
                i -= 1
            else:
                error += res
        errors.append(error/M)
    
    # print(errors)
    plt.xlabel("data distortion variance (Gaussian)")
    plt.ylabel("Average error of estimation (Frobenius)")
    plt.title("Test camera callibration using ellipses on erroneous data")
    plt.plot(np.arange(min_s, max_s, step_size), errors, 'ro')
    plt.show()


def get_con_points(i, s):
    x = -np.linspace(-5,5,500)
    y = np.linspace(-5,5,500)
    x,y = np.meshgrid(x,y)

    ats = [1, 2, 3]
    bts = [6, 4, 2]

    A = ats[i]
    C = bts[i]

    Z = A* x**2 + C*y**2 - 1
    c = plt.contour(x, y, (Z), [0], colors='blue')
    #plt.plot([],[],label = "Real", color = 'blue')
    x3 = []
    y3 = []
    for v in c.collections[0].get_paths():
        v = v.vertices
        x3.extend(v[:,0])
        y3.extend(v[:,1])


    x2 = [x3[i] + np.random.normal(scale = s) for i in range(len(x3))]
    y2 = [y3[i] + np.random.normal(scale = s) for i in range(len(y3))]

    # plt.show()
    plt.close()
    indices = random.sample(range(len(x2)), 200)

    xt = []
    yt = []

    for i in indices:
        xt.append(x2[i])
        yt.append(y2[i])


    return [np.array([x2[k], y2[k], 0, 1]) for k in range(len(x2))]

def con_fit(img_points):
    X = np.array([[img_points[i][0]**2, img_points[i][0]*img_points[i][1], img_points[i][1]**2, img_points[i][0]*img_points[i][2], img_points[i][1] * img_points[i][2], img_points[i][2]**2] for i in range(len(img_points))])

    i = np.array([[1], [1], [1], [1], [1], [1]])
    M = (np.linalg.inv(X.T @ X) @ i)

    try:
        M = (np.linalg.inv(X.T @ X) @ i)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("singular matrix")
            return (False, 'a')
        else:
            print("other problem")
            quit()

    b = M/(i.T @ M)
    # print(b.T)
    fit = b.T

    x2 = np.array([x[0]/x[2] for x in img_points])
    y2 = np.array([x[1]/x[2] for x in img_points])
    #start2 = time.time()
    fit_old_2 = fit
    grad1 = grad_f(fit_old_2[0],x2,y2)
    l_rate = 0.01
    fit_new_2 = fit_old_2 - l_rate * grad1
    iterations = 1

    while np.linalg.norm(fit_old_2 - fit_new_2) > 0.001:
        #print("Error: ", abs(f(fit_old[0]) - f(fit_new[0])))
        l_rate = (fit_new_2 - fit_old_2) @ (grad_f(fit_new_2[0],x2,y2) - grad_f(fit_old_2[0],x2,y2)).T / np.linalg.norm(grad_f(fit_new_2[0],x2,y2) - grad_f(fit_old_2[0],x2,y2)) ** 2
        l_rate = abs(l_rate[0,0])
        fit_old_2 = fit_new_2
        grad1 = grad_f(fit_old_2[0],x2,y2)
        fit_new_2 = fit_old_2 - l_rate * grad1
        iterations += 1
        #print("Goodness fit: ", f(fit_new_2[0]))
    
    fit_new_2 = fit_new_2 / np.sum(fit_new_2)
    # print(fit_new_2)

    return (True, b.T[0])

# print(get_con_points(2, .002))

# Applies conic homography estimation on one random homography and tests the correctness of its estimate using the Frobenius norm
# The data of the image after the gets a random error added to it
def test_cal(n, s):

    # Generate a random camera matrix P with intrinsic matrix K
    preK = randIntrinsic()
    R = randRot(3)
    t = np.array([[0],[0],[1]]) #(np.random.rand(3,1))
    Rt = np.hstack((R, t))
    P = preK @ Rt

    con =  [np.array([[1, 0, 0], [0, 6, 0], [0, 0, -1]]), np.array([[2, 0, 0], [0, 4, 0], [0, 0, -1]]), np.array([[3, 0, 0], [0, 2, 0], [0, 0, -1]])]

 
    homographies = []
    
    # Loop to get n homographies generated from n different virtual pictures of a randomly rotated and translated set of ellipses

    i = 0
    while i < n:
        # Generate a random rotation and translation
        R = randRot(3)
        t = np.array([[0],[0],[1]]) #(np.random.rand(3,1))


        # Randomly rotate and translate our chessboard
        Rt = np.vstack((np.hstack((R, t)), [0,0,0,1]))

        # param = [np.random.uniform(-3,3) for _ in range(6)]
        # param = [x / sum(param) for x in param]
        # A, B, C, D, E, F = param
       

        #con = np.array([[A, B/2, 0, D/2], [B/2, C, 0, E/2],  [0, 0, 0, 0], [D/2, E/2, 0, F, 0]])
        Q_inv = []
        w = 0
        for k in range(3):
            data = get_con_points(k, s)
            d3_plot(data)
            d3_plot(Rt @ x for x in data)
            image_data = [P @ Rt @ x for x in data]

            d2_plot(image_data)
            ris, param = con_fit(image_data)
            A = param[0]
            B = param[1]
            C = param[2]
            D = param[3]
            E = param[4]
            F = param[5]
            if ris and 4 * A * C - B**2 >= 0:
                
                
                #d2_plot_PARM(A, B, C, D, E, F)

                mat = np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])
                mat = (mat / np.linalg.det(mat))*np.linalg.det(con[k])

                Q_inv.append(np.linalg.inv(mat))
            else:
                print("Not a good rotation for a picture!")
                # d2_plot_PARM(A, B, C, D, E, F)
                w = 1
                break
            

        if w == 1:
            continue

        H = get_H(con, Q_inv)

        # Option to print the chessboard, its image, and the backprojection under our estimated H
        #big_plot(objpointsXY, imgpoints, [np.linalg.inv(H) @ X for X in imgpoints])

        big_plot([[x[0]/x[3], x[1]/x[3], 1] for x in data], image_data, [np.linalg.inv(H.T) @ x for x in image_data])  

        homographies.append(H)

        i += 1
    
    # Using Zhang's method, get the intrinsic matrix using homographies
    K = get_intrinsic(homographies, n)

    #print("Pre K = ", preK)
    #print("K = ", K)

    # Return the Frobenius error
    return np.linalg.norm( preK -  K)


print("Frob error K: ", test_cal(5, 0))