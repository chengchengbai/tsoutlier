# Time series outlier detection

An efficient non-parametric unsupervised time series anomaly detection algorithm

一个简单高效的非参无监督时间序列异常检测算法，根据Rob J Hyndman用R语言编写的forecast包中的异常检测方法改编而来

主要原理：将时序通过supersmooth的方法进行平滑，得到平滑后的数据，然后求其余与原始数据的偏差，然后对偏差运用Boxplot的方法得到异常点。算法还能对季节性数据做相应的季节性调整。
