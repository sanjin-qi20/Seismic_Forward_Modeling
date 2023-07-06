"""
============================
# -*- coding: utf-8 -*-
# Time    : 2023/3/30 20:58
# Author  : Qisx
# FileName: Clustel4Seismic.py
# Software: PyCharm
===========================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

# 将matplotlib字体设置为Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# 写一个标准化化函数
def standardize(x):
    return (x - x.mean()) / x.std()


seismic = pd.read_csv('seismic.csv', sep=',', encoding='utf-8')
seismic = seismic.values
# seismic = standardize(seismic)
x = []

for i in range(0, 498, 5):
    for j in range(0, 500, 20):
        x.append([i, seismic[i][j]])
X = np.array(x)

plt.figure(figsize=(8, 5))
model = AgglomerativeClustering(n_clusters=7, linkage='ward')
model.fit(X)
yhat = model.fit_predict(X)
# plt.subplot(2,2,2)
plt.title('AgglomerativeClustering', fontsize=18)

clusters = np.unique(yhat)
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 1], X[row_ix, 0])
    plt.ylim(500, 0)
    plt.yticks(np.linspace(0, 500, 5), np.linspace(400, 560, 5), fontsize=18)
    plt.xticks(fontsize=18)
plt.show()
plt.figure()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
Z = linkage(X, 'ward')
dendrogram(Z)
plt.show()

plt.figure(figsize=(8, 5))
model = KMeans(n_clusters=8)
model.fit(X)
yhat = model.fit_predict(X)
# plt.subplot(2,2,3)
plt.title('K-means', fontsize=18)
clusters = np.unique(yhat)
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 1], X[row_ix, 0])
    plt.ylim(500, 0)
    plt.yticks(np.linspace(0, 500, 5), np.linspace(400, 560, 5), fontsize=18)
    plt.xticks(fontsize=18)
plt.show()

plt.figure(figsize=(8, 5))
model = DBSCAN(eps=30, min_samples=200)
model.fit(X)
yhat = model.fit_predict(X)
# plt.subplot(2,2,4)
plt.title('DBSCAN', fontsize=18)
clusters = np.unique(yhat)
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 1], X[row_ix, 0])
    plt.ylim(500, 0)
    plt.yticks(np.linspace(0, 500, 5), np.linspace(400, 560, 5), fontsize=18)
    plt.xticks(fontsize=18)
plt.show()

plt.figure(figsize=(8, 5))
model = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_neighbors=10)
model.fit(X)
yhat = model.fit_predict(X)
# plt.subplot(2,2,4)
plt.title('SpectralClustering', fontsize=18)
clusters = np.unique(yhat)
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 1], X[row_ix, 0])
    plt.ylim(500, 0)
    plt.yticks(np.linspace(0, 500, 5), np.linspace(400, 560, 5), fontsize=18)
    plt.xticks(fontsize=18)
plt.show()
