import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from matplotlib.ticker import MaxNLocator

# 加载图像
original = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\original_image.png")
gaussian_noisy = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy2.png")
sp_noisy = cv2.imread(r"G:\FPGA\isp_raw_denosie\isp_raw_denosie\noisy22.png")

# 定义降噪方法
def apply_denoise_methods(noisy_img):
    """应用所有降噪方法"""
    results = {}
    
    # 均值滤波
    results['mean'] = cv2.blur(noisy_img, (3, 3))
    
    # 中值滤波
    results['median'] = cv2.medianBlur(noisy_img, 3)
    
    # 方框滤波 (归一化)
    results['box'] = cv2.boxFilter(noisy_img, -1, (3, 3), normalize=True)
    
    # 双边滤波
    results['bilateral'] = cv2.bilateralFilter(noisy_img, 15, 75, 75)
    
    # 高斯滤波
    results['gaussian'] = cv2.GaussianBlur(noisy_img, (5, 5), 0)
    
    return results

# 融合降噪方法
def fuse_denoise(noisy_img, methods=None):
    """融合多种降噪方法的结果"""
    if methods is None:
        methods = apply_denoise_methods(noisy_img)
    
    # 方法1：加权融合（基于方法特性分配权重）
    weights = {
        'mean': 0.15,
        'median': 0.30,  # 椒盐噪声效果好的方法权重更高
        'box': 0.15,
        'bilateral': 0.25,  # 保边效果好的方法
        'gaussian': 0.15
    }
    
    fused = np.zeros_like(noisy_img, dtype=np.float32)
    for method, img in methods.items():
        fused += weights.get(method, 0) * img.astype(np.float32)
    
    # 方法2：中值融合（对异常值更鲁棒）
    stack = np.stack([methods['mean'], methods['median'], 
                      methods['bilateral'], methods['gaussian']], axis=-1)
    median_fused = np.median(stack, axis=-1).astype(np.uint8)
    
    # 组合两种融合结果
    final_fused = (0.7 * fused + 0.3 * median_fused.astype(np.float32)).astype(np.uint8)
    
    return final_fused, methods

# 计算指标
def compute_metrics(original, denoised):
    """计算图像质量指标"""
    # 动态计算窗口大小
    win_size = min(original.shape[:2])
    if win_size % 2 == 0:
        win_size -= 1
    
    psnr = compare_psnr(original, denoised)
    ssim = compare_ssim(original, denoised, multichannel=True, 
                        win_size=win_size, channel_axis=2)
    mse = compare_mse(original, denoised)
    
    return {'PSNR': psnr, 'SSIM': ssim, 'MSE': mse}

# 可视化结果
def visualize_results(original, noisy, denoised_methods, fused_img, noise_type):
    """可视化降噪结果和指标"""
    # 创建子图布局
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.suptitle(f'Denoising Results Comparison ({noise_type} Noise)', fontsize=16)
    
    # 显示原始图像和噪声图像
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Noisy Image ({noise_type})')
    axes[0, 1].axis('off')
    
    # 显示各个降噪方法的结果
    methods = list(denoised_methods.keys())
    for i, method in enumerate(methods):
        row, col = divmod(i+2, 4)
        ax = axes[row, col]
        ax.imshow(cv2.cvtColor(denoised_methods[method], cv2.COLOR_BGR2RGB))
        ax.set_title(method.capitalize())
        ax.axis('off')
    
    # 显示融合结果
    axes[1, 3].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('Fused Denoise')
    axes[1, 3].axis('off')
    
    # 计算指标
    metrics = {}
    metrics['Noisy'] = compute_metrics(original, noisy)
    for method in methods:
        metrics[method] = compute_metrics(original, denoised_methods[method])
    metrics['Fused'] = compute_metrics(original, fused_img)
    
    # 保存可视化图像
    plt.tight_layout()
    plt.savefig(f'G:\\denoise_comparison_{noise_type.lower()}.png', dpi=120)
    plt.close()
    
    return metrics

# 生成指标图表
def plot_metrics(metrics, noise_type):
    """绘制不同方法的性能指标图表"""
    methods = list(metrics.keys())
    
    # 创建三个子图
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Denoising Performance Metrics ({noise_type} Noise)', fontsize=16)
    
    # PSNR 图表
    psnr_values = [metrics[method]['PSNR'] for method in methods]
    axs[0].bar(methods, psnr_values, color='skyblue')
    axs[0].set_title('PSNR Comparison')
    axs[0].set_ylabel('PSNR (dB)')
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))
    for i, v in enumerate(psnr_values):
        axs[0].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # SSIM 图表
    ssim_values = [metrics[method]['SSIM'] for method in methods]
    axs[1].bar(methods, ssim_values, color='lightgreen')
    axs[1].set_title('SSIM Comparison')
    axs[1].set_ylabel('SSIM')
    for i, v in enumerate(ssim_values):
        axs[1].text(i, v + 0.005, f"{v:.3f}", ha='center')
    
    # MSE 图表
    mse_values = [metrics[method]['MSE'] for method in methods]
    axs[2].bar(methods, mse_values, color='salmon')
    axs[2].set_title('MSE Comparison')
    axs[2].set_ylabel('MSE')
    for i, v in enumerate(mse_values):
        axs[2].text(i, v + 5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'G:\\denoise_metrics_{noise_type.lower()}.png', dpi=120)
    plt.close()

# 主处理流程
def process_noise(noisy_img, noise_type):
    """处理特定噪声类型的完整流程"""
    # 应用融合降噪
    fused_img, denoised_methods = fuse_denoise(noisy_img)
    
    # 可视化结果并获取指标
    metrics = visualize_results(original, noisy_img, denoised_methods, fused_img, noise_type)
    
    # 绘制指标图表
    plot_metrics(metrics, noise_type)
    
    # 保存融合结果
    cv2.imwrite(f"G:\\fused_denoise_{noise_type.lower()}.png", fused_img)
    
    # 打印融合结果指标
    print(f"\n融合降噪对{noise_type}噪声的指标:")
    print(f"PSNR: {metrics['Fused']['PSNR']:.2f}, "
          f"SSIM: {metrics['Fused']['SSIM']:.3f}, "
          f"MSE: {metrics['Fused']['MSE']:.1f}")
    
    return metrics

# 处理高斯噪声
print("处理高斯噪声...")
gaussian_metrics = process_noise(gaussian_noisy, "Gaussian")

# 处理椒盐噪声
print("\n处理椒盐噪声...")
sp_metrics = process_noise(sp_noisy, "Salt & Pepper")

# 打印所有方法在两种噪声上的指标对比
def print_comparison(metrics_dict, metric_name):
    """打印各方法在不同噪声下的指标对比"""
    print(f"\n{metric_name}对比:")
    print(f"{'Method':<12} {'Gaussian':<12} {'Salt & Pepper':<12} {'Avg':<12}")
    print("-" * 45)
    
    methods = [m for m in metrics_dict['Gaussian'] if m != 'Noisy']
    for method in methods:
        g_val = metrics_dict['Gaussian'][method][metric_name]
        s_val = metrics_dict['Salt & Pepper'][method][metric_name]
        avg = (g_val + s_val) / 2
        print(f"{method:<12} {g_val:<12.3f} {s_val:<12.3f} {avg:<12.3f}")

# 打印三种指标的对比
print_comparison({'Gaussian': gaussian_metrics, 'Salt & Pepper': sp_metrics}, 'PSNR')
print_comparison({'Gaussian': gaussian_metrics, 'Salt & Pepper': sp_metrics}, 'SSIM')
print_comparison({'Gaussian': gaussian_metrics, 'Salt & Pepper': sp_metrics}, 'MSE')

# 保存所有结果图
print("\n所有结果已保存至G盘根目录:")
print("- 降噪效果对比图 (denoise_comparison_*.png)")
print("- 指标对比图表 (denoise_metrics_*.png)")
print("- 融合降噪结果图 (fused_denoise_*.png)")