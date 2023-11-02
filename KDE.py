import math
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 训练集目录
train_dir = ".\\2023\\train"
# 测试集目录
test_dir = ".\\2023\\test"


def load_image(dir_name):
    """
    加载目录dir_name下的所有图片，并将所有图片的信息转换成数组存储在images中
    :param dir_name:存放图片的目录
    :return: images:所有图像array类型数据，是一个四维数组，他的shape为(高度，宽度，通道数，图片总数)
    """
    images_name = os.listdir(dir_name)      # 读取dir_name下的所有文件（经人工检查该目录下没有子目录），结果存入列表images中
    images_name.sort()                      # 对列表images按照名称字符顺序进行排序
    images_total = len(images_name)         # 获取图片总数目

    ''' 此处切换数据的尺寸(不同数据集有不同的尺寸) '''
    if train_dir[2] == '2':
        images = np.empty((576, 768, 3, images_total))      # 创建空数组，用于储存图像信息
        print("2023数据集")
    else:
        images = np.empty((540, 960, 3, images_total))      # 创建空数组，用于储存图像信息
        print("mydata数据集")

    for i, img_name in enumerate(images_name):
        """
        遍历所有图片，依次将每张图片转换成numpy数组，并将所有图片存放在（576，768，3，len(图片数量)）大小的数组类型images中
        """
        try:  # 使用pillow库打开图片，失败则报错
            img_open = Image.open(os.path.join(dir_name, img_name))
        except Exception as e:
            print(f"无法读取图像：{e}")

        image = np.asarray(img_open)    # 将每张图片转换成numpy中的数组，方便进行矩阵运算
        images[:, :, :, i] = image      # 将图片依次存入images中

    return images


def epanechnikov_kde(train, test, h: float = 100):
    """
        Epanechnikov核作为核函数下的核密度估计
    :param train:训练集，类型array
    :param test:测试集，类型array
    :param h:带宽参数
    :return:将核密度估计值
    """
    factor1 = 15 / (8 * math.pi)  # 简化带宽矩阵时，核密度估计函数前的系数
    factor2 = 1 / (N * pow(h, dim))  # Ke(x)前的系数
    f = np.zeros((height, width))  # 二维数组，存放每个像素的概率
    for i in range(height):
        for j in range(width):
            """
                从上到下、从左到右，依次遍历每个像素，计算每个像素的概率
            """
            for n in range(N):
                ksm = 1  # 存放多个通道结果的乘积（用于多维变量的核函数）
                for d in range(dim):
                    """
                        多维变量的Ks可以通过每个维度下单变量核函数的乘积运算来表示。
                        在此处维度为 3 ，所以分别计算三个维度下的Ks1，Ks2，Ks3，求得乘积,
                        即 Ks = K1 * k2 * k3
                    """
                    ksm *= 1 - ((test[i, j, d] - train[i, j, d, n]) / h) ** 2
                    if ksm < 0:
                        """
                            单维度的EP核函数的公式中写到，平方项大于1时 Ke(x) = 0
                            即（1-平方项）< 0 时，Ke(x) = 0
                            因为 Ke(x) 与 此代码中的ksm 只是相差一个系数
                            所以只要令 ksm = 0 ，即可实现Ke(x) = 0
                        """
                        ksm = 0
                f[i, j] = f[i, j] + ksm  # 对N张图片的[i,j]处像素点的概率密度求和
            f[i, j] = factor1 * factor2 * f[i, j]  # [i,j]处像素点的概率密度 * 系数，得到核密度估计值

    return f  # 将核密度估计值返回


def triangular_kde(train, test, h: float = 100):
    """
        三角核
    :param train:训练集，类型array
    :param test:测试集，类型array
    :param h:带宽参数
    :return:将核密度估计值
    """
    factor = 1 / (N * pow(h, dim))  # Ke(x)前的系数
    f = np.zeros((height, width))  # 二维数组，存放每个像素的概率
    for i in range(height):
        for j in range(width):
            """
                从上到下、从左到右，依次遍历每个像素，计算每个像素的概率
            """
            for n in range(N):
                ksm = 1  # 存放多个通道结果的乘积（用于多维变量的核函数）
                for d in range(dim):
                    """
                        多维变量的Ks可以通过每个维度下单变量核函数的乘积运算来表示。
                        在此处维度为 3 ，所以分别计算三个维度下的Ks1，Ks2，Ks3，求得乘积,
                        即 Ks = K1 * k2 * k3
                    """
                    ksm *= 1 - abs((test[i, j, d] - train[i, j, d, n]) / h)
                    if ksm < 0:
                        """
                            三角核数学表达式为
                                |x|<=1时, K(x)=1-|x|
                                |x|>1时, K(x)=0
                            即 (1-|x|) < 0 时，K(x) = 0
                            因为 Ke(x) 与 此代码中的ksm 只是相差一个系数
                            所以只要令 ksm = 0 ，即可实现Ke(x) = 0
                        """
                        ksm = 0
                f[i, j] = f[i, j] + ksm  # 对N张图片的[i,j]处像素点的概率密度求和
            f[i, j] = factor * f[i, j]  # [i,j]处像素点的概率密度 * 系数，得到核密度估计值

    return f  # 将核密度估计值返回


train_data = load_image(train_dir)              # 加载训练集
print(train_data.shape)                         # 打印数据集的shape
height, width, dim, N = train_data.shape        # 数据各个维度的含义为 (高度，宽度，通道数，图片总数)
test_data = load_image(test_dir)[:, :, :, 0]    # 加载测试集

# f = epanechnikov_kde(train_data, test_data)   # 计算ep核的核密度估计值
'''  此处设置 9 个不同的h  '''
h_list = np.linspace(30, 270, 9)  # 设置9个不同的带宽
''' 此处设置 9 个不同的阈值 '''
threshold_value_list = np.linspace(0.2, 7.0, 9)  # [start, ..., stop]
threshold_value_list = threshold_value_list * math.pow(10, -7)

f_ij = np.zeros((height, width, 9))         # 用来存放不同 h 时的核密度估计
g_threshold = np.zeros((height, width, 9))  # 用来存放相同阈值，不同 h 下的二值图像
g_h = np.zeros((height, width, 9))          # 存放固定h，不同阈值下的二值图像
for i, h in enumerate(h_list):
    """
        计算不同h下的核密度估计值得到f_ij
        并将其二值化得到g_ij
    """
    ''' 此处可选择使用ep核或者三角核 '''
    # f_ij[:, :, i] = epanechnikov_kde(train_data, test_data, h)
    f_ij[:, :, i] = triangular_kde(train_data, test_data, h)

    ''' 固定一个h，观察不同阈值下的二值图像有什么不同（对应于第三张图） '''
    if h == 90:
        for j, threshold_value in enumerate(threshold_value_list):
            g_h[:, :, j] = np.where(f_ij[:, :, i] < threshold_value, 1, 0)

    ''' 固定一个阈值，观察不同h下的二值图像有什么不同（对应于第二张图） '''
    threshold_value = 2 * math.pow(10, -7)                              # 设置阈值（切一刀）
    g_threshold[:, :, i] = np.where(f_ij[:, :, i] < threshold_value, 1, 0)  # 使用np.where对图像(数组)的值进行更改

    print(f"{i}\th = {h}已完成，进度{(i + 1) * 11}%")                         # 显示进度，不至于盯着控制台很无聊


"""================================ 绘图 ================================"""
# 画 3d 图形
x, y = np.meshgrid(np.arange(width), np.arange(height))
fig, ax = plt.subplots(3, 3)  # 定义3*3的画布
for i in range(3):
    for j in range(3):
        """
            依次在画布上绘图
        """
        k = i * 3 + j
        ax[i, j].set_axis_off()                                     # 去除刻度
        ax[i, j] = fig.add_subplot(331 + k, projection='3d')        # 在第k个位置绘图
        ax[i, j].plot_surface(x, y, f_ij[:, :, k], cmap='viridis')  # 绘制核密度估计值的三维表面
        ax[i, j].set_title(f'h = {h_list[k]}', fontsize=12)         # 设置标题
plt.show()

# 画二值图像（同阈值下变h）
fig2, ax2 = plt.subplots(3, 3)  # 定义3*3的画布
for i in range(3):
    for j in range(3):
        """
            依次在画布上绘图
        """
        k = i * 3 + j
        ax2[i, j].set_axis_off()                                # 去除刻度
        ax2[i, j].imshow(g_threshold[:, :, k], cmap='gray')     # 绘制灰度图
        ax2[i, j].set_title(f'h = {h_list[k]}', fontsize=12)    # 设置标题
plt.show()

# 画二值图像（同h下变阈值）
fig3, ax3 = plt.subplots(3, 3)  # 定义3*3的画布
for i in range(3):
    for j in range(3):
        """
            依次在画布上绘图
        """
        k = i * 3 + j
        ax3[i, j].set_axis_off()                                # 去除刻度
        ax3[i, j].imshow(g_h[:, :, k], cmap='gray')             # 绘制灰度图
        ax3[i, j].set_title(f'threshold = {threshold_value_list[k]}', fontsize=12)  # 设置标题
plt.show()
