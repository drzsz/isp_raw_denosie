import numpy as np
import cv2

def gaussian_noise(image, mean, var):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise

    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out

image = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\original_image.png")

noisy1 = gaussian_noise(image, mean=0, var=0.01)
noisy2 = gaussian_noise(image, mean=0.1, var=0.01)
noisy3 = gaussian_noise(image, mean=0, var=0.2)

h1 = np.hstack([image, noisy1])
h2 = np.hstack([noisy2, noisy3])
v = np.vstack([h1, h2])

# 保存高斯噪声后的图像
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy1.png", noisy1)
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy2.png", noisy2)
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy3.png", noisy3)

# 保存合成的图像
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\gaussian_image.png", v)

cv2.imshow('out', v)
cv2.waitKey()