import matplotlib.pyplot as plt
import numpy as np

img1 = np.zeros([200, 300, 3]) # 全0，黑色图像
plt.imshow(img1)
plt.show()

img2 = np.full([200, 300, 3], 255) # 全0，黑色图像
plt.imshow(img2)
plt.show()

img3 = np.full([200, 300, 3], 128) # 全0，黑色图像
plt.imshow(img3)
plt.show()

img = plt.imread(r"F:\Maker\Learn_Systematically\6_Deep_learning\3_Convolutional_Neural_Networks_CNN\Meeting_at_the_Peak.jpg")
plt.imshow(img)
plt.show()