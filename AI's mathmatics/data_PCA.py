import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv('Prb_data/Prb01data02.csv', header=None)

# 데이터 센터링
mean_vector = d.mean(axis=1)
d_centered = d.sub(mean_vector, axis=0)

# 공분산 행렬 계산
cov_matrix = np.cov(d_centered.values)

# SVD
U, s, Vt = np.linalg.svd(cov_matrix)

# 신호의 혼합 방식 A와 원래 신호 s(t) 추정
A_estimated = Vt.T
s_estimated = np.dot(A_estimated.T, d.values)

print("Estimated Mixing Matrix A:\n", A_estimated)
print("Estimated Original Signals s(t):\n", s_estimated)

# Plot the estimated signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(s_estimated[0, :])
plt.title('Estimated signal 1')

plt.subplot(2, 1, 2)
plt.plot(s_estimated[1, :])
plt.title('Estimated signal 2')

plt.tight_layout()
plt.show()