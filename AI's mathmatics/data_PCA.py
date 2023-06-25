import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read the data
d = pd.read_csv('Prb_data/Prb01data01.csv', header=None)
d = d.transpose()

# Initialize PCA
pca = PCA(n_components=2)
pca.fit(d)

# Recover the original signals (apply transform)
s_estimated = pca.transform(d)

# Get the mixing matrix
A_estimated = pca.components_.T

print("Estimated Mixing Matrix A:\n", A_estimated)
print("Estimated Original Signals s(t):\n", s_estimated.T)

# Plot the estimated signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(s_estimated[:, 0])
plt.title('Estimated signal 1')

plt.subplot(2, 1, 2)
plt.plot(s_estimated[:, 1])
plt.title('Estimated signal 2')

plt.tight_layout()
plt.show()