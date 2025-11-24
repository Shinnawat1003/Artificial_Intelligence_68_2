import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# ========================
# 1. PARAMETERS (ตามสไลด์)
# ========================

mean_A = np.array([0.0, 0.0])
cov_A  = np.array([[0.10, 0.00],
                   [0.00, 0.75]])

mean_B = np.array([3.2, 0.0])
cov_B  = np.array([[0.75, 0.00],
                   [0.00, 0.10]])

N = 300    # จำนวนข้อมูลต่อคลาส


# ========================
# 2. GENERATE DATA
# ========================

ptsA = np.random.multivariate_normal(mean_A, cov_A, N)
ptsB = np.random.multivariate_normal(mean_B, cov_B, N)


# ========================
# 3. BAYES DECISION BOUNDARY
# ========================

# grid (ช่วงกว้างเพื่อเห็นเส้นชัด)
x = np.linspace(-2, 6.5, 400)
y = np.linspace(-3, 3, 400)
xx, yy = np.meshgrid(x, y)

# combine grid into 2D points
pos = np.dstack((xx, yy))

# likelihoods
pA = multivariate_normal(mean_A, cov_A).pdf(pos)
pB = multivariate_normal(mean_B, cov_B).pdf(pos)

# priors = 0.5 each
posterior_A = pA * 0.5
posterior_B = pB * 0.5

# decision boundary = where posterior_A = posterior_B
boundary = posterior_A - posterior_B


# ========================
# 4. PLOT
# ========================

plt.figure(figsize=(10,6))

plt.scatter(ptsA[:,0], ptsA[:,1], c='red', alpha=0.5, label='A')
plt.scatter(ptsB[:,0], ptsB[:,1], c='blue', alpha=0.5, label='B')

# contour where difference = 0  → decision boundary
plt.contour(xx, yy, boundary, levels=[0], colors='purple', linewidths=2)

plt.grid(True)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("Bayes Decision Boundary Between Class A and Class B")
plt.show()
