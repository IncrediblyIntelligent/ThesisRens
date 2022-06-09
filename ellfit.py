import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.transform import warp
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
import math
from matplotlib.patches import Ellipse

plt.figure()

x = -np.linspace(-4,4,1000)
y = np.linspace(-4,4,1000)
x,y = np.meshgrid(x,y)

A, B, C, D, E, F = (3, 1, 1.5, 2, 1, 1)
A, B, C, D, E, F = (2, 2, 1, .3, 0, -1)
A, B, C, D, E, F = [np.random.rand()*12 -6 for _ in range(6)]
F = -1



Z = A* x**2 + B*x*y + C*y**2 + D*x + E*y + F
c = plt.contour(x, y, (Z), [0], colors='black')
x3 = []
y3 = []
for v in c.collections[0].get_paths():
    v = v.vertices
    x3.extend(v[:,0])
    y3.extend(v[:,1])


# v = c.collections[0].get_paths()[0].vertices
# x3 = v[:,0]
# y3 = v[:,1]

x2 = [x3[i] + .1*np.random.skellam(.1) for i in range(len(x3))]
y2 = [y3[i] + .1*np.random.poisson(.1) for i in range(len(y3))]

indices = np.random.choice(range(len(x2)), 49)

x2 = [x2[i] for i in indices]
y2 = [y2[i] for i in indices]
# plt.xlim([-1.5,1.5])
# plt.ylim([-11.5,-8.5])

#print(x)
#print(y)

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
plt.show()