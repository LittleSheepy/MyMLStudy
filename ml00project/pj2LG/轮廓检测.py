import cv2
import os

# 定义函数，输入原始图片路径和保存路径，输出画好轮廓的图片
def draw_contours(img_path, save_path):
    # 读取图片
    img = cv2.imread(img_path)
    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.blur(img_gray, (3, 3))
    for i in range(10):
        img_blur = cv2.blur(img_blur, (3, 3))
    # 边缘检测
    edges = cv2.Canny(img_blur, 30, 100)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 画轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    # 保存图片
    cv2.imwrite(save_path, img)

# 定义函数，输入原始文件夹路径和保存文件夹路径，对原始文件夹中的所有图片进行轮廓检测并保存到保存文件夹中
def draw_contours_folder(input_folder, output_folder):
    # 遍历文件夹
    for file_name in os.listdir(input_folder):
        # 判断是否为图片文件
        if file_name.endswith('.bmp') or file_name.endswith('.png'):
            # 构造原始图片路径和保存路径
            img_path = os.path.join(input_folder, file_name)
            save_path = os.path.join(output_folder, file_name)
            # 调用画轮廓函数
            draw_contours(img_path, save_path)
if __name__ == '__main__':
    root_dir = r"D:\04DataSets\ningjingLG\02ZangWu\LateralPollution/"
    input_folder = root_dir + 'img/'
    output_folder = root_dir + 'contours/'
    draw_contours_folder(input_folder, output_folder)
