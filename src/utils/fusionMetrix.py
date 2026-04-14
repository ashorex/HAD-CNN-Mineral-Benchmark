import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 填入你图片中的具体数值
# 每一行代表真实类别 (True)，每一列代表预测类别 (Predicted)
cm_data = np.array([[360,   0, 180,   0],[ 50, 360, 130,   0],[  0,   0, 840,  60],
    [  0,   0,   0, 540]
])

# 2. 设置画布大小
plt.figure(figsize=(7, 5))

# 3. 绘制热力图
# annot=True: 显示数字
# fmt='g': 正常显示数字（不使用科学计数法）
# cmap='Blues': 使用和你原图一样的蓝色渐变色系
# cbar=True: 显示右侧的颜色条
sns.heatmap(cm_data, annot=True, fmt='g', cmap='Blues', cbar=True,
            xticklabels=[0, 1, 2, 3],
            yticklabels=[0, 1, 2, 3])

# 4. 添加标题和坐标轴标签
plt.title('Confusion Matrix - HDA-CNN')
plt.ylabel('True')
plt.xlabel('Predicted')

# 5. 保存为高清矢量图（适合插入PPT，无限放大不模糊）
plt.savefig("HDA_CNN_Confusion_Matrix.svg", format="svg", bbox_inches='tight')
plt.show()