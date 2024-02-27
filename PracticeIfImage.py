import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread('Pogi (2).jpg', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Apply different filters
blur = cv.blur(img_gray, (5, 55))
Gblur = cv.GaussianBlur(img_gray, (55, 5), 0)
Median = cv.medianBlur(img_gray, 5)
Bblur = cv.bilateralFilter(img_gray, 20, 200, 200)
sobelxy = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
filtered_image_xy = cv.convertScaleAbs(sobelxy)
edges = cv.Canny(img_gray, 50, 150)
laplacian_edges = cv.Laplacian(img_rgb, cv.CV_64F)

# Plot images in a single window
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(331)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('on')

# Grayscale Image
plt.subplot(332)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')
plt.axis('on')

# Blurred Image
plt.subplot(333)
plt.imshow(blur, cmap='gray')
plt.title('Blur')
plt.axis('on')

# Gaussian Blurred Image
plt.subplot(334)
plt.imshow(Gblur, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('on')

# Median Blurred Image
plt.subplot(335)
plt.imshow(Median, cmap='gray')
plt.title('Median Blur')
plt.axis('on')

# Bilateral Filtered Image
plt.subplot(336)
plt.imshow(Bblur, cmap='gray')
plt.title('Bilateral Filter')
plt.axis('on')

# Sobel XY
plt.subplot(337)
plt.imshow(filtered_image_xy, cmap='gray')
plt.title('Sobel XY')
plt.axis('on')

# Canny Edges
plt.subplot(338)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('on')

# Laplacian Edges
plt.subplot(339)
plt.imshow(laplacian_edges, cmap='gray')
plt.title('Laplacian Edges')
plt.axis('on')

plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()