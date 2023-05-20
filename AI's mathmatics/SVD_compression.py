import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# PIL Image 객체 생성
img = Image.open('Myimage.jpg')

# 이미지 scaling
img = np.array(img)/255

img_k = np.mean(img, axis=2)

# 원본 흑백 이미지
plt.figure()
plt.imshow(img_k, cmap='gray')
plt.title('Original grayscale image')
plt.show()

# 원본 RGB 이미지
plt.figure()
plt.imshow(img)
plt.title('Original RGB image')
plt.show()

# 흑백 이미지의 SVD
u, d, v = np.linalg.svd(img_k)

k = 50    # 주성분 개수를 설정합니다.

Ai = np.zeros_like(img_k)
for i in range(k):
    Ai += d[i] * np.outer(u[:, i], v[i, :])    # 압축

# 압축된 흑백 이미지
plt.figure()
plt.imshow(Ai, cmap='gray')
plt.title('Compressed grayscale image')
plt.show()

# 흑백 이미지를 압축하는 데 사용된 주성분 저장
ustore = u[:, :k]
vstore = v[:k, :]
dstore = d[:k]

NumUncomp = np.size(img_k)
NumComp = np.size(ustore) + np.size(vstore) + np.size(dstore)

# 원본 이미지와 압축된 이미지의 픽셀 개수를 출력합니다.
print(f'압축 전 픽셀 개수 = {NumUncomp}')
print(f'압축 후 픽셀 개수 = {NumComp}')