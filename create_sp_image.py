import numpy as np
import cv2
import random

def sp_noise(image, amount):
    output = image.copy()
    threshold = 1 - amount

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdm = random.random()
            if rdm < amount:
                output[i][j] = 0
            elif rdm > threshold:
                output[i][j] = 255

    return output

image = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\original_image.png")

# 添加不同量的椒盐噪声
noisy1 = sp_noise(image, amount=0.05)
noisy2 = sp_noise(image, amount=0.1)
noisy3 = sp_noise(image, amount=0.2)

# 拼接图片
h1 = np.hstack([image, noisy1])
h2 = np.hstack([noisy2, noisy3])
v = np.vstack([h1, h2])

# 保存椒盐噪声后的图像
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy11.png", noisy1)
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy22.png", noisy2)
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy33.png", noisy3)

# 保存合成的图像
cv2.imwrite(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\sp_image.png", v)

cv2.imshow('out', v)
cv2.waitKey()
