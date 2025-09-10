import cv2
import numpy as np
import time
import os

def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法（修正通道顺序问题）
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 正确分离BGR通道
    b, g, r = cv2.split(img)
    b_avg = cv2.mean(b)[0]
    g_avg = cv2.mean(g)[0]
    r_avg = cv2.mean(r)[0]
    
    # 求各个通道所占增益
    k = (b_avg + g_avg + r_avg) / 3
    kb = k / max(b_avg, 1e-5)
    kg = k / max(g_avg, 1e-5)
    kr = k / max(r_avg, 1e-5)
    
    # 应用增益
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    
    # 合并通道返回
    return cv2.merge([b, g, r])

def white_balance_2(img_input):
    '''
    完美反射白平衡（优化性能，使用向量化操作）
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    
    # 向量化计算RGB和
    sum_ = b.astype(np.int32) + g.astype(np.int32) + r.astype(np.int32)
    
    # 计算阈值（使用百分比）
    ratio = 0.01
    threshold = np.percentile(sum_, 100 * (1 - ratio))
    
    # 计算高亮区域平均值
    mask = sum_ >= threshold
    if np.any(mask):
        avg_b = np.mean(b[mask])
        avg_g = np.mean(g[mask])
        avg_r = np.mean(r[mask])
    else:
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    
    # 防止除零
    avg_b = max(avg_b, 1e-5)
    avg_g = max(avg_g, 1e-5)
    avg_r = max(avg_r, 1e-5)
    
    # 向量化调整像素值
    max_value = 255.0
    b = np.clip(b * max_value / avg_b, 0, 255).astype(np.uint8)
    g = np.clip(g * max_value / avg_g, 0, 255).astype(np.uint8)
    r = np.clip(r * max_value / avg_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b, g, r])

def white_balance_3(img):
    '''
    灰度世界假设（优化性能，使用向量化操作）
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 向量化操作替代循环
    b, g, r = cv2.split(img.astype(np.float32))
    
    # 计算平均值
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
    k = (b_avg + g_avg + r_avg) / 3
    
    # 计算增益
    kb = k / max(b_avg, 1e-5)
    kg = k / max(g_avg, 1e-5)
    kr = k / max(r_avg, 1e-5)
    
    # 应用增益并限制范围
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

def white_balance_4(img):
    '''
    基于图像分析的偏色检测及颜色校正方法（修复数学错误）
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    b, g, r = cv2.split(img.astype(np.float32))
    m, n = b.shape
    
    # 向量化计算
    I_r_2 = r ** 2
    I_b_2 = b ** 2
    
    # 计算统计值
    sum_I_r_2 = np.sum(I_r_2)
    sum_I_b_2 = np.sum(I_b_2)
    sum_I_r = np.sum(r)
    sum_I_b = np.sum(b)
    sum_I_g = np.sum(g)
    
    max_I_r = np.max(r)
    max_I_b = np.max(b)
    max_I_g = np.max(g)
    max_I_r_2 = np.max(I_r_2)
    max_I_b_2 = np.max(I_b_2)
    
    # 防止奇异矩阵
    det_b = sum_I_b_2 * max_I_b - sum_I_b * max_I_b_2
    det_r = sum_I_r_2 * max_I_r - sum_I_r * max_I_r_2
    
    if abs(det_b) < 1e-5 or abs(det_r) < 1e-5:
        return img
    
    # 求解线性方程组
    u_b = (sum_I_g * max_I_b - max_I_g * sum_I_b) / det_b
    v_b = (sum_I_b_2 * max_I_g - max_I_b_2 * sum_I_g) / det_b
    
    u_r = (sum_I_g * max_I_r - max_I_g * sum_I_r) / det_r
    v_r = (sum_I_r_2 * max_I_g - max_I_r_2 * sum_I_g) / det_r
    
    # 应用校正
    b_corrected = np.clip(u_b * I_b_2 + v_b * b, 0, 255)
    r_corrected = np.clip(u_r * I_r_2 + v_r * r, 0, 255)
    
    return cv2.merge([b_corrected, g, r_corrected]).astype(np.uint8)

def white_balance_5(img,max_gain=2.0):
    '''
    动态阈值算法（优化性能，使用向量化操作），增加增益限制
    :param img: cv2.imread读取的图片数据
    :param max_gain: 最大允许增益值，防止过度曝光
    :return: 返回的白平衡结果图片数据
    '''
    b, g, r = cv2.split(img)
    
    # 转换到YCrCb空间
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, u, v = cv2.split(yuv_img)
    
    # 计算U和V的平均值和绝对偏差
    avl_u = np.mean(u)
    avl_v = np.mean(v)
    avl_du = np.mean(np.abs(u - avl_u))
    avl_dv = np.mean(np.abs(v - avl_v))
    
    # 创建选择掩码
    con_u = np.sign(avl_u)
    con_v = np.sign(avl_v)
    radio = 0.5
    mask = (np.abs(u - (avl_u + avl_du * con_u)) < radio * avl_du) | \
           (np.abs(v - (avl_v + avl_dv * con_v)) < radio * avl_dv)
    
    # 计算亮点的阈值
    y_masked = np.where(mask, y, 0)
    yhistogram = np.bincount(y_masked.flatten().astype(int), minlength=256)
    ysum = np.sum(yhistogram)
    
    if ysum == 0:
        return img
    
    cum_hist = np.cumsum(yhistogram[::-1])[::-1]
    key = np.argmax(cum_hist > 0.1 * ysum)
    
    # 计算高亮区域的平均值
    highlight_mask = (y >= key) & mask
    if np.any(highlight_mask):
        avl_r = np.mean(r[highlight_mask])
        avl_g = np.mean(g[highlight_mask])
        avl_b = np.mean(b[highlight_mask])
    else:
        avl_r, avl_g, avl_b = np.mean(r), np.mean(g), np.mean(b)
    
    # 防止除零
    avl_b = max(avl_b, 1e-5)
    avl_g = max(avl_g, 1e-5)
    avl_r = max(avl_r, 1e-5)
    
    # 应用校正，但限制最大增益
    max_y = 255.0
    gain_b = min(max_y / avl_b, max_gain)
    gain_g = min(max_y / avl_g, max_gain)
    gain_r = min(max_y / avl_r, max_gain)
    
    b = np.clip(b * gain_b, 0, 255).astype(np.uint8)
    g = np.clip(g * gain_g, 0, 255).astype(np.uint8)
    r = np.clip(r * gain_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b, g, r])

# 创建输出目录的函数
def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# 保存图片的函数
def save_image(image, filename, output_dir):
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")

# 主程序
if __name__ == "__main__":
    # 指定输出目录
    output_dir = r"G:\FPGA\isp\AWB"
    
    # 创建输出目录
    output_dir = create_output_dir(output_dir)
    
    # 读取图片（修改为您的输入图片路径）
    input_image_path = r"G:\FPGA\isp\isp_raw_denosie\isp_raw_denosie\original_image.png"
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_image_path}")
    
    # 保存原始图片
    save_image(img, "original.png", output_dir)
    
    # 定义处理方法
    methods = [
        ("simple_mean", white_balance_1),
        ("perfect_reflection", white_balance_2),
        ("gray_world", white_balance_3),
        ("color_deviation", white_balance_4),
        ("dynamic_threshold", white_balance_5)
    ]
    
    # 处理并保存每种方法的图片
    processed_images = [img]
    processed_names = ["Original"]
    
    for name, func in methods:
        start = time.time()
        result = func(img)
        end = time.time()
        print(f"{name} executed in {end - start:.4f} seconds")
        
        # 保存处理后的图片
        save_image(result, f"{name}.png", output_dir)
        
        # 准备堆叠显示
        processed_images.append(result)
        processed_names.append(name.replace('_', ' ').title())
    
    # 创建堆叠对比图
    # 调整大小以适应显示
    resized_images = []
    for img in processed_images:
        if img.shape[0] > 800 or img.shape[1] > 1200:
            scale = min(800 / img.shape[0], 1200 / img.shape[1])
            resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        else:
            resized = img
        resized_images.append(resized)
    
    # 添加文字标签
    labeled_images = []
    for img, name in zip(resized_images, processed_names):
        # 添加标签背景
        label_bg = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        label_bg[:, :] = (40, 40, 40)  # 深灰色背景
        cv2.putText(label_bg, name, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 将标签和图片合并
        labeled = np.vstack([label_bg, img])
        labeled_images.append(labeled)
    
    # 垂直堆叠结果
    img_stack = np.vstack(labeled_images)
    
    # 保存对比图
    comparison_path = os.path.join(output_dir, "comparison.png")
    cv2.imwrite(comparison_path, img_stack)
    print(f"Saved comparison image: {comparison_path}")
    
    # 显示结果
    cv2.imshow('White Balance Comparison', img_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"All results saved in directory: {os.path.abspath(output_dir)}")