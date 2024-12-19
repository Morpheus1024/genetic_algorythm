import matplotlib.pyplot as plt

import numpy as np
import cv2
#wczytaj obrazy

img1 = cv2.imread('Szablon 1_1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('Szablon 1_2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread('Szablon 2_1.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread('Szablon 2_2.png')
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img5 = cv2.imread('Szablon 3_1.png')
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
img6 = cv2.imread('Szablon 3_2.png')
img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)
img7 = cv2.imread('Szablon 4_1.png')
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)
img8 = cv2.imread('Szablon 4_2.png')
img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(4, 2, figsize=(10, 15))

axs[0, 0].imshow(img1)
axs[0, 0].axis('off')
axs[0, 0].set_title('Szablon 1 - przebieg ewolucji')

axs[0, 1].imshow(img2)
axs[0, 1].axis('off')
axs[0, 1].set_title('Szablon 1 - trasa końcowa')

axs[1, 0].imshow(img3)
axs[1, 0].axis('off')
axs[1, 0].set_title('Szablon 2 - przebieg ewolucji')

axs[1, 1].imshow(img4)
axs[1, 1].axis('off')
axs[1, 1].set_title('Szablon 2 - trasa końcowa')

axs[2, 0].imshow(img5)
axs[2, 0].axis('off')
axs[2, 0].set_title('Szablon 3 - przebieg ewolucji')

axs[2, 1].imshow(img6)
axs[2, 1].axis('off')
axs[2, 1].set_title('Szablon 3 - trasa końcowa')

axs[3, 0].imshow(img7)
axs[3, 0].axis('off')
axs[3, 0].set_title('Szablon 4 - przebieg ewolucji')

axs[3, 1].imshow(img8)
axs[3, 1].set_title('Szablon 4 - trasa końcowa')
axs[3, 1].axis('off')


plt.tight_layout()
plt.show()