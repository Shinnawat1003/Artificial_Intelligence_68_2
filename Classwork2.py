import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. กำหนด mean และ covariance
# =========================

# จุดกลางวงกลมแดง (class a)
mean1 = np.array([-3.0, 5.0])

# จุดกลางวงกลมน้ำเงิน (class b)
mean2 = np.array([ 3.0, 5.0])

# ใช้ covariance เป็น identity -> ก้อนจะเป็นวงกลม
cov = np.array([[1.0, 0.0],
                [0.0, 1.0]])

# =========================
# 2. สุ่มจุดจาก Gaussian
# =========================
N = 200  # จำนวนจุดต่อคลาส

pts1 = np.random.multivariate_normal(mean1, cov, size=N)
pts2 = np.random.multivariate_normal(mean2, cov, size=N)

plt.figure(figsize=(8, 6))

plt.scatter(pts1[:, 0], pts1[:, 1],
            marker='o', s=50, alpha=0.5,
            color='red', label='a')

plt.scatter(pts2[:, 0], pts2[:, 1],
            marker='o', s=50, alpha=0.5,
            color='blue', label='b')

# ---  เพิ่มเส้นแกน Y (x = 0) แบบทึบ สีแดง ---
plt.axvline(x=0, color='red', linewidth=3)

plt.axis('equal')

plt.xlabel('X')
plt.ylabel('Y')

# กำหนดช่วงแกน
plt.xlim(-6, 6)
plt.ylim(-1, 10)

plt.legend()
plt.grid(True)

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.title("Two Gaussian Blobs at (-3,5) and (3,5) with Red Y-axis Divider")
plt.show()
