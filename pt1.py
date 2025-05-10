import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def add_salt_pepper_noise(img, prob):
    noisy = img.copy()
    rows, cols = noisy.shape  
    num_noise = int(prob * rows * cols)

    for _ in range(num_noise):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        if random.random() < 0.5:
            noisy[i, j] = 0
        else:
            noisy[i, j] = 255

    return noisy


image = cv2.imread('zaina.jpg')
image = cv2.resize(image, (512, 512))
cv2.imshow('Original', image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

watermarked = image.copy()
h, w = image.shape[:2]
x = random.randint(0, w - 300)
y = random.randint(50, h - 50)

watermarked = cv2.putText(watermarked, "ZAINA ;) 12112550",  (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA, False)

cv2.imwrite('watermarked.jpg', watermarked)

cv2.imshow('Watermarked Image', watermarked)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("dimensions:", image.shape)
print("- mean:", np.mean(image))
print("- min:", np.min(image))
print("- max:", np.max(image))

c = random.uniform(0.4, 2.0)
bright = cv2.convertScaleAbs(image, alpha=c, beta=0)
cv2.imwrite('brightness_modified.jpg', bright)
cv2.imshow('Brightness Modified', bright)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(bright.ravel(), 256, [0, 256])
plt.title("Histogram of Brightness Modified Image")
plt.xlabel("gray level")
plt.ylabel("count")
plt.show()

gray_bright = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray_bright)
cv2.imwrite('equalized.jpg', equalized)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.xlabel("gray level")
plt.ylabel("count")
plt.show()
noisy_img = add_salt_pepper_noise(gray_image, 0.02)
cv2.imwrite('noisy_gray.jpg', noisy_img)
cv2.imshow('Noisy Grayscale Image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

mean_filtered = cv2.blur(noisy_img, (3, 3))
cv2.imwrite('mean_filtered_gray.jpg', mean_filtered)
cv2.imshow('Mean Filtered Grayscale', mean_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

median_filtered = cv2.medianBlur(noisy_img, 3)
cv2.imwrite('median_filtered_gray.jpg', median_filtered)
cv2.imshow('Median Filtered Grayscale', median_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(median_filtered, -1, filter)
cv2.imwrite('sharpened_gray.jpg', sharpened)
cv2.imshow('Sharpened Grayscale Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
