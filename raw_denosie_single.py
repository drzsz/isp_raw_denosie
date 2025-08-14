import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse

original = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\original_image.png")
gaussian_png = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy3.png")
sp_png = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy22.png")

def compare(ori, gaussian_denosing, sp_denosing):

    # ori为原图（未被噪声污染的图），gaussian_denosing为经过高斯噪声污染的图通过去噪算法去噪后的图；sp_denosing为经过椒盐噪声污染的图通过去噪算法去噪后的图
    gaussian_psnr = compare_psnr(ori, gaussian_denosing)
    sp_psnr = compare_psnr(ori, sp_denosing)

    gaussian_ssim = compare_ssim(ori, gaussian_denosing, multichannel=True)
    sp_ssim = compare_ssim(ori, sp_denosing, multichannel=True)

    gaussian_mse = compare_mse(ori, gaussian_denosing)
    sp_mse = compare_mse(ori, sp_denosing)

    return gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse

#均值滤波

gaussian_blur = cv2.blur(gaussian_png, (3, 3))
sp_blur = cv2.blur(sp_png, (3, 3))
h1 = np.hstack([gaussian_png, gaussian_blur])
h2 = np.hstack([sp_png, sp_blur])
v = np.vstack([h1, h2])
cv2.imwrite(r"G:\mean_filter.png", v)
cv2.imshow('out', v)
cv2.waitKey()

gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse = compare(original, gaussian_blur, sp_blur)
print('均值滤波对高斯噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(gaussian_psnr, gaussian_ssim, gaussian_mse))
print('均值滤波对椒盐噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(sp_psnr, sp_ssim, sp_mse))

'''
#中值滤波

gaussian_blur = cv2.medianBlur(gaussian_png, 3)
sp_blur = cv2.medianBlur(sp_png, 3)
h1 = np.hstack([gaussian_png, gaussian_blur])
h2 = np.hstack([sp_png, sp_blur])
v = np.vstack([h1, h2])
cv2.imwrite(r"G:\median_filter.png", v)
cv2.imshow('out', v)
cv2.waitKey(1000)
cv2.destroyAllWindows()  # 关闭窗口


gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse = compare(original, gaussian_blur, sp_blur)
print('中值滤波对高斯噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(gaussian_psnr, gaussian_ssim, gaussian_mse))
print('中值滤波对椒盐噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(sp_psnr, sp_ssim, sp_mse))

#方框滤波

gaussian_blur = cv2.boxFilter(gaussian_png, -1, (3, 3), normalize=0)
sp_blur = cv2.boxFilter(sp_png, -1, (3, 3), normalize=0)
h1 = np.hstack([gaussian_png, gaussian_blur])
h2 = np.hstack([sp_png, sp_blur])
v = np.vstack([h1, h2])
cv2.imwrite(r"G:\box_filter.png", v)
cv2.imshow('out', v)
cv2.waitKey()



gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse = compare(original, gaussian_blur, sp_blur)
print('方框滤波对高斯噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(gaussian_psnr, gaussian_ssim, gaussian_mse))
print('方框滤波对椒盐噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(sp_psnr, sp_ssim, sp_mse))


#双边滤波
gaussian_blur = cv2.bilateralFilter(gaussian_png, 25, 100, 100)
sp_blur = cv2.bilateralFilter(sp_png, 25, 100, 100)
h1 = np.hstack([gaussian_png, gaussian_blur])
h2 = np.hstack([sp_png, sp_blur])
v = np.vstack([h1, h2])
cv2.imwrite(r"G:\bilateral_filter.png", v)
cv2.imshow('out', v)
cv2.waitKey()


gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse = compare(original, gaussian_blur, sp_blur)
print('双边滤波对高斯噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(gaussian_psnr, gaussian_ssim, gaussian_mse))
print('双边滤波对椒盐噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(sp_psnr, sp_ssim, sp_mse))

#高斯滤波

gaussian_blur = cv2.GaussianBlur(gaussian_png, (5, 5), 0, 0)
sp_blur = cv2.GaussianBlur(sp_png, (5, 5), 0, 0)
h1 = np.hstack([gaussian_png, gaussian_blur])
h2 = np.hstack([sp_png, sp_blur])
v = np.vstack([h1, h2])
cv2.imwrite(r"G:\gaussian_filter.png", v)
cv2.imshow('out', v)
cv2.waitKey()


gaussian_psnr, sp_psnr, gaussian_ssim, sp_ssim, gaussian_mse, sp_mse = compare(original, gaussian_blur, sp_blur)
print('高斯滤波对高斯噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(gaussian_psnr, gaussian_ssim, gaussian_mse))
print('高斯滤波对椒盐噪声污染图像去噪后的指标为:PSNR:{},SSIM:{},MSE:{}'.format(sp_psnr, sp_ssim, sp_mse))'

'''
