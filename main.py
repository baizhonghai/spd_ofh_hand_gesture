from util.file_process import read_files_to_list, calculate_slide_window, preprocess_dataset
from util.gaussian_calculate import calculate_gaussian_embedding_for_dataset,calculate_gaussian_embedding_for_gaussian_model


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    # 读文件就先简单的到读到dict_list里边就好了
    dataset_dict = read_files_to_list('./datasets/ETH-80')

    # 图片预处理，返回图片数据集
    preprocessed_dataset = preprocess_dataset(dataset_dict, resize_x=256, resize_y=256)

    # 每个元素而言，用滑动窗口切割成小块。
    window_matrix = calculate_slide_window(x=256, y=256, x_num=9, y_num=9)
    # 对每一类别下的元素进行解读
    gaussian_dict = calculate_gaussian_embedding_for_dataset(preprocessed_dataset, window_matrix)

    # 计算高斯模型之间的协方差。
    spd_dict = calculate_gaussian_embedding_for_gaussian_model(gaussian_dict)
    # 再次构造高斯嵌入。
    '''到此得到了80个SPD矩阵进行分类'''

    # 尝试1，直接用log Euclidean 转换成Euclidean域内的东西然后进行bestk or pca降为分类，
    # 尝试2，再找个滑动窗口的尺寸，而后构造第二个SPD矩阵组，将1&2的结果融合后bestk. 这里可以增加窗口的尺寸数量，至多4个吧。
    # 尝试3，将SPD矩阵丢到SPDNet中去处理。
    # 尝试4，将多尺寸的SPD组合成一个大的矩阵，丢到SPDNet处理。
    # 尝试5，调整SPDNet并行处理这些不同维度的SPD,而后连接起来分类。

# 前边进行代码转换的时候参考matlab的结果和实现，搞起来。争取这周把这个尝试完，如果不好使再吸收别的文章，还有教授可能也会给些新的建议。
