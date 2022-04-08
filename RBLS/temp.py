from sklearn import preprocessing
import numpy as np # 1. 标准化：去均值，方差规模化
# 创建一组特征数据，每一行表示一个样本，每一列表示一个特征
# Standardization标准化:将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值维0，方差为1.
# 标准化的原因在于如果有些特征的方差过大，则会主导目标函数从而使参数估计器无法正确地去学习其他特征。# 标准化的过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）。
# 在sklearn.preprocessing中提供了一个scale的方法，可以实现以上功能。
x = np.array([[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]])
# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
x_scale = preprocessing.scale(x, axis=1)
print(x_scale)