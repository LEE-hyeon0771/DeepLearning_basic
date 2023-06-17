import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('Prb_data/Prb02data01.csv', header=None)

# 첫 번째, 두 번째, 세 번째 데이터 선택
data1 = df.iloc[:, 0]
data2 = df.iloc[:, 1]
data3 = df.iloc[:, 2]
# 그래프 생성
plt.figure(figsize=(10,6))
plt.plot(data1)
plt.ylim(-0.12, 0.06)
plt.yticks(np.arange(-0.12, 0.07, 0.02))
plt.xticks(np.arange(0, 451, 50))
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage data from the first column of Prb02data01.csv')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(data2)
plt.ylim(-0.12, 0.06)
plt.yticks(np.arange(-0.12, 0.07, 0.02))
plt.xticks(np.arange(0, 451, 50))
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage data from the second column of Prb02data01.csv')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(data3)
plt.ylim(-0.12, 0.06)
plt.yticks(np.arange(-0.12, 0.07, 0.02))
plt.xticks(np.arange(0, 451, 50))
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage data from the third column of Prb02data01.csv')
plt.show()



