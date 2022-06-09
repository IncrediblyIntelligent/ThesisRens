from tkinter import N
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
import math
from matplotlib.patches import Ellipse

x = -np.linspace(-6,6,100)
y = np.linspace(-6,6,100)
x,y = np.meshgrid(x,y)

A, B, C, D, E, F = (2, 2, 1, 4, 0, -1)

Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F
c = plt.contour(x, y, (Z), [0], colors='black')
plt.show()
x3 = []
y3 = []
for v in c.collections[0].get_paths():
    v = v.vertices
    x3.extend(v[:,0])
    y3.extend(v[:,1])



x2 = [x3[i] + .1*np.random.normal(scale = .01) for i in range(len(x3))]
y2 = [y3[i] + .1*np.random.normal(scale = .01) for i in range(len(y3))]

X = np.array([[x2[i]**2, x2[i]*y2[i], y2[i]**2, x2[i], y2[i], 1] for i in range(len(x2))])

i = np.array([[1], [1], [1], [1], [1], [1]])
M = (np.linalg.inv(X.T @ X) @ i)

b = M/(i.T @ M)
b = -b/b[5]
print("b = ", b)
print("original =", np.array([A, B, C, D, E, F]))
print(np.linalg.norm(b.T - np.array([A, B, C, D, E, F])))

plt.scatter(x2, y2,s=12)

Z2 = b[0]* x**2 + b[1]* x*y + b[2]* y**2 + b[3] * x + b[4] * y + b[5]

plt.contour(x, y, Z2, [0], colors='red')
plt.title("Original")
#plt.show()

def simulate(A,B,C,D,E, s, F = -1):
    Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F
    c = plt.contour(x, y, (Z), [0], colors='black')
    x3 = []
    y3 = []
    for v in c.collections[0].get_paths():
        v = v.vertices
        x3.extend(v[:,0])
        y3.extend(v[:,1])



    x2 = [x3[i] + .1*np.random.normal(scale = s) for i in range(len(x3))]
    y2 = [y3[i] + .1*np.random.normal(scale = s) for i in range(len(y3))]

    X = np.array([[x2[i]**2, x2[i]*y2[i], y2[i]**2, x2[i], y2[i], 1] for i in range(len(x2))])
    X2 = np.array([[x2[i]**2, x2[i]*y2[i], y2[i]**2, x2[i], y2[i]] for i in range(len(x2))])

    i = np.array([[1], [1], [1], [1], [1], [1]])
    i2 = np.array([[1]] * len(x2))
    M = (np.linalg.inv(X.T @ X) @ i)
    M2 = np.linalg.inv(X2.T @ X2)

    b1 = M/(i.T @ M)
    b1 = -b1/b1[5]
    b2 = (M2 @ X2.T) @ i2

    return b1, b2

b1, b2 = simulate(A,B,C,D,E,0.1)
print(b1)
print(b2)

# n = 100
# MSES = []
# scales = np.arange(0,3,0.1)
# for s in scales:
#     print(s)
#     MSE = 0
#     for _ in range(n):
#         b = simulate(A,B,C,D,E , s)
#         MSE += np.linalg.norm(b.T - np.array([A, B, C, D, E, F]))/n 
#     MSES.append(MSE)

# plt.cla()

# plt.plot(scales, MSES)
# plt.show()
# print(MSES)

