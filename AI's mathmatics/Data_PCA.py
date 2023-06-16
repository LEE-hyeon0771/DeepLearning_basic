import pandas as pd
import numpy as np

# 데이터 읽기
d = []
with open('Prb_data/Prb01data01.csv', 'r') as f:
    d = f.read().split()
d[0] = list(map(float, d[0].split(",")))
d[1] = list(map(float, d[1].split(",")))

df = pd.DataFrame(d)

# 데이터 센터링
mean_vector = df.mean(axis=1)
df_centered = df.sub(mean_vector, axis=0)

# 공분산 행렬 계산
cov_matrix = np.cov(df_centered.values)

# SVD 수행
U, s, Vt = np.linalg.svd(cov_matrix)

# 신호의 혼합 방식 A와 원래 신호 s(t) 추정
A_estimated = Vt
s_estimated = U @ np.diag(s)

print("Estimated Mixing Matrix A:\n", A_estimated)
print("Estimated Original Signals s(t):\n", s_estimated)

