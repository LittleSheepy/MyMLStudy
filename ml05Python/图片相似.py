from skimage.metrics import structural_similarity as ssim
import cv2
import os
import imagehash
from PIL import Image
import cv2
def calculate_complexity_similarity(img1str, img2str):
    # 加载两张图片
    img1 = cv2.imread(img1str)
    img2 = cv2.imread(img2str)

    # 将图片转换为灰度图像
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 检测特征点和描述符
    kp1, des1 = orb.detectAndCompute(gray_img1, None)
    kp2, des2 = orb.detectAndCompute(gray_img2, None)

    # 创建暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)
    similarity=0.0
    # 根据特征点匹配结果计算相似度
    if len(matches) > 0:
        similarity = sum([match.distance for match in matches]) / len(matches)
        # print('图片相似度为：', similarity)
    # else:
    #     print('未找到匹配的特征点')
        # 调用函数进行图片相似度计算,计算简单的图片相似度
    return similarity
def calculate_histogram_similarity(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 计算直方图
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # 归一化直方图
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # 比较直方图
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    if similarity<0.6:
        similarity=calculate_complexity_similarity(img1_path, img2_path)

    return similarity
def compare_images(imageA, imageB):
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    return score

def compareHist(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM) between two images
    score = cv2.compareHist(img1_gray, img2_gray, cv2.HISTCMP_BHATTACHARYYA)
    return score

def hash(img1, img2):
    # 计算哈希值
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    # 判断哈希值相似度
    similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    return similarity

def find_dir():
    max_score = 0
    max_file = ""
    for file in os.listdir(img_dir):
        file_path = img_dir + file
        # img1 = cv2.imread(img_path1)
        # img21 = cv2.imread(file_path)
        img1 = Image.open(img_path1)
        img2 = Image.open(file_path)
        try:
            score = calculate_histogram_similarity(img_path1, file_path)
        except Exception:
            pass
        if score >  max_score:
            max_score = score
            max_file = file
            print("max_score: ", max_score," max_file:", max_file)
    print("result max_score: ", max_score," max_file:", max_file)



if __name__ == '__main__':
    print("")
    img_dir = r"D:\test1/"
    img_path1 = r"d:/book.jpg"
    find_dir()


