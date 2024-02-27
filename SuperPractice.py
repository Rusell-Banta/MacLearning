import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Pogi (2).jpg', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

blur = cv.blur(img_rgb, (5, 55))
Gblur = cv.GaussianBlur(img_rgb, (55, 5), 5)
Median = cv.medianBlur(img_rgb, 5)
Bblur = cv.bilateralFilter(img_rgb, 20, 200, 200)
sobelxy = cv.Sobel(img_gray, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
filtered_image_xy = cv.convertScaleAbs(sobelxy)
edges = cv.Canny(img_gray, 50, 150)
laplacian_edges = cv.Laplacian(img_rgb, cv.CV_64F)

plt.figure(figsize=(12,10))

plt.subplot(331)
plt.imshow(img_rgb)
plt.text(400,100, 'Original', color = 'Black', fontsize = 20)

plt.subplot(332)
plt.imshow(img_gray, cmap = 'gray')
plt.text(400,100, 'Grayscale', color = 'Black', fontsize = 20)

plt.subplot(333)
plt.imshow(blur)
plt.text(400,100, 'Blur', color = 'Black', fontsize = 20)

plt.subplot(334)
plt.imshow(Gblur)
plt.text(400,100, 'Gaussian Blur', color = 'Black', fontsize = 20)

plt.subplot(335)
plt.imshow(Median)
plt.text(400, 100, 'Median Blur', color = 'Black', fontsize = 20)

plt.subplot(336)
plt.imshow(Bblur)
plt.text(400, 100, 'Bilateral Filter', color = 'Black', fontsize = 20)

plt.subplot(337)
plt.imshow(filtered_image_xy, cmap='gray')
plt.text(400, 100, 'Sobel XY', color = 'White', fontsize = 20)

plt.subplot(338)
plt.imshow(edges, cmap='gray')
plt.text(400, 100, 'Canny Edges', color = 'White', fontsize = 20)

plt.subplot(339)
plt.imshow(laplacian_edges, cmap='gray')
plt.text(400, 100, 'Laplacian Edges', color = 'White', fontsize = 20)



plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

