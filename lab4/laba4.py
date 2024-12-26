import numpy as np
import cv2

def setup(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height//2
    pad_w = kernel_width//2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                          mode="constant", constant_values=0)
    result = np.zeros_like(image)
    return image_height, image_width, padded_image, result, kernel_height, kernel_width

def erosion(image, kernel):
    height, width, padded_image, result, kernel_height, kernel_width = setup(image, kernel)
    for i in range(height):
        for j in range(width):
            pr = padded_image[i:i + kernel_height, j:j + kernel_width]
            if np.all(pr[kernel == 1] == 1):
                result[i, j] = 1
    return result

def dilation(image, kernel):
    height, width, padded_image, result, kernel_height, kernel_width = setup(image, kernel)
    for i in range(height):
        for j in range(width):
            pr = padded_image[i:i + kernel_height, j:j + kernel_width]
            if np.any(pr[main_kernel == 1] > 0):
                result[i, j] = 1
    return result

def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)

def prep(a):
    return a * 225

main_image = (cv2.imread("start.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
main_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], np.uint8)
cv2.imwrite("erosion.png", prep(erosion(main_image, main_kernel)))
cv2.imwrite("dilation.png", prep(dilation(main_image, main_kernel)))
cv2.imwrite("opening.png", prep(opening(main_image, main_kernel)))
cv2.imwrite("closing.png", prep(closing(main_image, main_kernel)))