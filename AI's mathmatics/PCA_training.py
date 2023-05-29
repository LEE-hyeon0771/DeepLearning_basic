import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# 이미지 데이터 불러오기
data = []
for i in range(1, 21):  # 20개의 이미지
    img = Image.open(f'training data set/Image{i}.png').convert('L').resize((100, 100))
    img_data = np.array(img).flatten()
    data.append(img_data)

# 이미지 갯수
n_images = len(data)

# 모든 이미지에 대해 Eigen-vectors와 근사 이미지를 계산하고 시각화
fig, ax = plt.subplots(6, n_images, figsize=(6 * n_images, 25))

pca = PCA(n_components=n_images)  # 모든 가능한 eigen-vectors
pca.fit(data)

# 각 이미지
for j in range(n_images):
    # 원본 이미지 표시
    ax[0, j].imshow(data[j].reshape((100, 100)), cmap='gray')

    # Eigen-vectors 시각화
    for i in range(2):
        ax[i + 1, j].imshow(pca.components_[i].reshape((100, 100)), cmap='gray')

    # 주성분의 갯수에 따른 근사 이미지 표시
    for i, n_components in enumerate([5, 10, 20]):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        approximated = pca.inverse_transform(transformed[j])
        ax[i + 3, j].imshow(approximated.reshape((100, 100)), cmap='gray')

# 새로운 이미지 로드 및 벡터화
# new_image_1.png: training data set에 없지만, 같은 종류의 사진
# new_image_2.png: training data set에 없지만, 다른 종류의 사진
new_images = []
new_img_names = ['new_image_1.png', 'new_image_2.png']
for img_name in new_img_names:
    img = Image.open(img_name).convert('L').resize((100, 100))
    img_data = np.array(img).flatten()
    new_images.append(img_data)

# 새로운 이미지의 원본과 근사 이미지 표시
fig2, ax2 = plt.subplots(3, len(new_img_names), figsize=(5 * len(new_img_names), 15))
for j, new_image in enumerate(new_images):
    transformed = pca.transform([new_image])
    approximated = pca.inverse_transform(transformed)

    ax2[0, j].imshow(new_image.reshape((100, 100)), cmap='gray')
    for i in range(2):
        ax2[i + 1, j].imshow(pca.components_[i].reshape((100, 100)), cmap='gray')

titles = ['Original Image', 'Eigen-vector 1', 'Eigen-vector 2', '5 components',
         '10 components', '20 components']

for i in range(6):
    ax[i, 0].set_title(titles[i])

new_image_titles = ['New Image', 'Eigen-vector 1', 'Eigen-vector 2']
for i in range(3):
    ax2[i, 0].set_title(new_image_titles[i])

plt.tight_layout()
plt.show()