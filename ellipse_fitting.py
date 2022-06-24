from tkinter import E, N
import random
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import csv

# helper function
def f(param,subX,subY):
    A,B,C,D,E,F = param
    som = 0
    for i in range(len(subX)):
        x = subX[i]
        y = subY[i]
        som += (A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F) ** 2 / ((2 * A * x + B * y + D) ** 2 + (B * x + 2 * C * y + E) ** 2)
    return som

# helper function
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

# helper function
def g(param, subX,subY):
    A,B,C,D,E,F = param
    i = random.randint(0, len(subX) - 1)
    x = subX[i]
    y = subY[i]
    return (A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F) ** 2 / ((2 * A * x + B * y + D) ** 2 + (B * x + 2 * C * y + E) ** 2)

# helper function
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

# simulates algebraic and gradient descent fitting on a random ellipse
# does one simulation, with variance s on data and n_points amount of randomly chosen points given for fitting
# returns whether fitting could be done succesful, along with the distances of the algebraic fit, gradient descent fit, and stochastic descent fit to the original parameters, in that order 
def one_sim(s, n_points, iter):
    x = -np.linspace(-3,3,500)
    y = np.linspace(-3,3,500)
    x,y = np.meshgrid(x,y)

    param = [np.random.uniform(-3,3) for _ in range(6)]
    param = [x / sum(param) for x in param]
    A, B, C, D, E, F = param

    Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F
    c = plt.contour(x, y, (Z), [0], colors='red')
    plt.plot([],[],label = "Real", color = 'red')
    x3 = []
    y3 = []
    for v in c.collections[0].get_paths():
        v = v.vertices
        x3.extend(v[:,0])
        y3.extend(v[:,1])

    if len(x3) < n_points:
        return (False,0,0,0)


    plt.clf()

    x2 = [x3[i] + np.random.normal(scale = s) for i in range(len(x3))]
    y2 = [y3[i] + np.random.normal(scale = s) for i in range(len(y3))]


    indices = range(len(x2))
    indices = random.sample(indices, n_points)

    xt = []
    yt = []

    for i in indices:
        xt.append(x2[i])
        yt.append(y2[i])

    x2 = xt
    y2 = yt

    X = np.array([[x2[i]**2, x2[i]*y2[i], y2[i]**2, x2[i], y2[i], 1] for i in range(len(x2))])

    i = np.array([[1], [1], [1], [1], [1], [1]])
    M = (np.linalg.inv(X.T @ X) @ i)

    try:
        M = (np.linalg.inv(X.T @ X) @ i)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("singular matrix")
            return (False, 'a', 'a', 'a')
        else:
            print("other problem")
            quit()

    b = M/(i.T @ M)

    original = np.array(param)

    fit = b.T

    fit_old = fit
    grad1 = grad_g(fit_old[0],x2,y2)
    l_rate = 0.001
    fit_new = fit_old - l_rate * grad1

    for i in range(n_points * 100):
        fit_old = fit_new
        grad1 = grad_g(fit_old[0],x2,y2)
        fit_new = fit_old - l_rate * grad1

    fit_old_2 = fit
    grad1 = grad_f(fit_old_2[0],x2,y2)
    l_rate = 0.01
    fit_new_2 = fit_old_2 - l_rate * grad1
    iterations = 1

    while np.linalg.norm(fit_old_2 - fit_new_2) > 0.0001:
        l_rate = (fit_new_2 - fit_old_2) @ (grad_f(fit_new_2[0],x2,y2) - grad_f(fit_old_2[0],x2,y2)).T / np.linalg.norm(grad_f(fit_new_2[0],x2,y2) - grad_f(fit_old_2[0],x2,y2)) ** 2
        l_rate = abs(l_rate[0,0])
        fit_old_2 = fit_new_2
        grad1 = grad_f(fit_old_2[0],x2,y2)
        fit_new_2 = fit_old_2 - l_rate * grad1
        iterations += 1

    end2 = time.time()

    fit_new = fit_new / np.sum(fit_new)
    fit_new_2 = fit_new_2 / np.sum(fit_new_2)

    norms = [np.linalg.norm(fit - original), np.linalg.norm(fit_new - original), np.linalg.norm(fit_new_2 - original)]
    print(norms[0], norms[2], norms[1])

    return (True, norms[0], norms[2], norms[1])

# make a test plot with M iterations, simulating one_sim on n_points points with variance from min_s to max_s with stepsize step_size
# returns the data of the simulated values
def test_plot(M, min_s, max_s, step_size, n_points):
    error_alg = []
    error_grad = []
    error_stoch = []
    k = -1
    for s in np.arange(min_s, max_s, step_size):
        k += 1
        error_alg.append(0)
        error_grad.append(0)
        error_stoch.append(0)
        i = 0
        while i < M:
            i += 1
            print("s, i = ", s, i)
            res, alg, grad, stoch = one_sim(s, n_points, i)
            # Test if calculation went well
            if(not res):
                i -= 1
            else:
                error_alg[k] += alg / M
                error_grad[k] += grad / M
                error_stoch[k] += stoch / M
    
    plt.clf()

    plt.xlabel("data error variance (normal)")
    plt.ylabel("MSE of the estimations")
    plt.title("Conic fitting on 200 erroneous data points.")

    plt.plot(np.arange(min_s, max_s, step_size), error_alg, 'ro', label= "avg. algebraic fitting error")
    plt.plot(np.arange(min_s, max_s, step_size), error_grad, 'bo', label= "avg. gradient fitting error")
    plt.plot(np.arange(min_s, max_s, step_size), error_stoch, 'go', label= "avg. stochastic fitting error")

    plt.legend(loc='upper left')
    plt.show()
    return (error_alg, error_grad, error_stoch)

# example of a test: N=1000 simulating variance from 0 to 0.1 with steps of size 0.01, giving the simulation 250 points of a random conic to estimate
# its parameters based on a subset of size 250 of its pixels. Returns a plot with the average correctness of algebraic fitting, gradient descent fitting
# stochastic descent fitting
test_plot(1,0,0.1,0.01,250)