"""
============================
# -*- coding: utf-8 -*-
# Time    : 2023/4/6 15:31
# Author  : Qisx
# FileName: ClusteDatas.py
# Software: PyCharm
===========================
"""
# k-means 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# 将matplotlib字体设置为Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# 定义数据集
X, _ = make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=4, n_clusters_per_class=2, random_state=4)
# 定义模型
model = KMeans(n_clusters=6)
# 模型拟合
model.fit(X)
# 为每个示例分配一个集群
yhat = model.predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
plt.figure(figsize=(8, 5))
for cluster in clusters:
# 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.title('K-means', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()


model=AgglomerativeClustering(n_clusters=6,linkage='ward')
model.fit(X)
yhat=model.fit_predict(X)
plt.figure(figsize=(8, 5))
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.title('AgglomerativeClustering', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()
plt.figure()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
Z = linkage(X)
dendrogram(Z)
plt.show()

model= DBSCAN(eps=0.9, min_samples=100, leaf_size=10)
model.fit(X)
yhat=model.fit_predict(X)
plt.figure(figsize=(8, 5))
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.title('DBSCAN', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()

model=SpectralClustering(n_clusters=6,affinity='nearest_neighbors',n_neighbors=10)
model.fit(X)
yhat=model.fit_predict(X)
plt.figure(figsize=(8, 5))
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.title('SpectralClustering', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()