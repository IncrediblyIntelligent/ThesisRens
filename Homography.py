import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy

#np.random.seed(seed = 42)

#Xr = np.array([[-3.717606428508024, -2.098691842634128, 1], [-4.051330903389136, -0.2665481082314658, 1], [-5.564586499719232, 0.5519316890835348, 1], [3.7646544542092357, -5.490092207877763, 1], [-4.6531195268561145, 5.509282654863158, 1], [-4.184196072973219, 1.1872094690601385, 1], [-4.020463552655932, -5.573979987310782, 1], [-0.7269066808713145, 4.228322668005173, 1], [-0.4074791019554773, 1.4250474664130044, 1], [-1.2390770252555665, -3.6863433061383324, 1], [-3.7302681089670378, 4.763129290548729, 1], [-2.320136864416186, -3.533910617538995, 1], [0.11515713424489604, 3.0312389427860182, 1], [-5.902537536125643, 2.0737511936813178, 1], [5.176905593020255, 0.19273818662426834, 1], [-5.154191059633652, 3.1059628313181857, 1], [5.404772830039828, 3.987859301515705, 1], [2.4436960552968507, 5.577237094245627, 1], [1.8876529820698362, -1.760570374896285, 1], [2.1030425079683823, -1.2800107362581867, 1]])

# n_data = 4
# std = 0.01

# Xr = np.array([[np.random.uniform(-6,6), np.random.uniform(-6,6), 1] for _ in range(n_data)])

# def randHom(n):
#     H = np.random.rand(n, n)
#     if np.linalg.det(H) != 0:
#         return H/H[2,2]
#     else:
#         return randHom(n)

# H = randHom(3)
# print(H)
# Yr = Xr @ H.T

# Xob = copy.deepcopy(Xr)
# Yob = copy.deepcopy(Yr)

# for i in range(len(Xob)):
#     Xob[i,0] += np.random.normal(scale = std)
#     Xob[i,1] += np.random.normal(scale = std)
#     Xob[i,2] += np.random.normal(scale = std)
#     Yob[i,0] += np.random.normal(scale = std)
#     Yob[i,1] += np.random.normal(scale = std)
#     Yob[i,2] += np.random.normal(scale = std)

# u, s, v = np.linalg.svd(Xob)

# X = np.array([np.append(Yob[i],Xob[i]) for i in range(len(Xob))])

# u2, s2, v2 = np.linalg.svd(X)
# #print(s)
# #print(s2)

# H_hat = np.linalg.inv(Xob.T @ Xob - s2[-1] ** 2 * np.identity(3)) @ Xob.T @ Yob
# print(H_hat.T)
# print(np.linalg.norm(H_hat.T - H))

n_data = 49
std = 0.01

Xr = np.array([[np.random.uniform(-6,6), np.random.uniform(-6,6), 1] * 3 for _ in range(n_data)])


print(Xr)

u, s, v = np.linalg.svd(Xr)
print(s)
