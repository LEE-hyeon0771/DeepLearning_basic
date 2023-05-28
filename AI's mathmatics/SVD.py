import numpy as np
import pandas as pd

# 데이터 로드
df = pd.read_csv('data_cp01.csv')
data = df.values

# 데이터 센터링
mean = np.mean(data, axis=0)
data_centered = data - mean

# 공분산 행렬 계산
cov_matrix = np.cov(data_centered.T)

# 공분산 행렬의 특잇값 분해
U, S, Vt = np.linalg.svd(cov_matrix)

# 첫 번째 주성분에 데이터 투영(2차원 -> 1차원 축소)
first_pc = U[:, 0]
projected_data = np.dot(data_centered, first_pc)

print(projected_data)