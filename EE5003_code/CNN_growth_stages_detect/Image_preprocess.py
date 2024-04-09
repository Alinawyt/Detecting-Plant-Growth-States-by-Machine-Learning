import cv2
import numpy as np
import os

# 定义生菜的颜色范围
# lower_green = np.array([30, 50, 50])  # 生菜的最小HSV值
# upper_green = np.array([85, 255, 255])  # 生菜的最大HSV值
# png0
lower_green = np.array([25, 50, 50])  # 生菜的最小HSV值
upper_green = np.array([85, 255, 255])  # 生菜的最大HSV值
# lower_green = np.array([30, 100, 50])  # 生菜的最小HSV值
# upper_green = np.array([85, 255, 255])  # 生菜的最大HSV值
# 待处理的文件夹路径
folder_path = '/Users/alina./CODE/EE5003/program_final/HogSvm/test2/2'
outputfolder_path = '/Users/alina./CODE/EE5003/program_final/HogSvm/test2/filter_2'
# 遍历文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 读取图像
        image = cv2.imread(os.path.join(folder_path, filename))

        # 将图像转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建生菜的掩码
        lettuce_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # 对生菜区域进行形态学处理，填充孔洞，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        lettuce_mask = cv2.morphologyEx(lettuce_mask, cv2.MORPH_CLOSE, kernel)

        # 获取背景区域
        background_mask = cv2.bitwise_not(lettuce_mask)

        # 创建白色背景图像
        white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)

        # 从原始图像中提取生菜和背景，并叠加到白色背景上
        lettuce = cv2.bitwise_and(image, image, mask=lettuce_mask)
        background = cv2.bitwise_and(white_background, white_background, mask=background_mask)
        result = cv2.add(lettuce, background)

        # 保存分离结果
        output_path = os.path.join(outputfolder_path, 'filter_' + filename)
        cv2.imwrite(output_path, result)

