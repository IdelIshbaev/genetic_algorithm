import numpy as np
import cv2 as cv
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = r'/Users/idel_isbaev/PycharmProjects/IAI_ass/images/img_01.jpg'
input_image = cv.imread(path)

plt.subplot(121)
plt.axis("off")
plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
plt.subplot(122)
plt.axis("off")
plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
plt.show()