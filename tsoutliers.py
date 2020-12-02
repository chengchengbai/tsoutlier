import numpy as np
import pandas as pd

from supersmoother.supersmoother import SuperSmoother


class SmootherAD:

    def __init__(self, seasadj=False):
        """
        :param seasadj: True or False, 是否需要季节性调整，对于明显的周期性数据，应设置为True
        """
        self.seasadj = seasadj
        self.smoother = SuperSmoother()

        self.data = None
        self.smoothed_data = None
        self.upper_limit, self.lower_limit = None, None
        self.outliers = None

    def detect(self, ts: pd.Series) -> np.ndarray:
        """
        :param ts: pd.Series类型，索引为时间
        :return outliers: numpy.ndarray 异常点的布尔索引，通过ts[outliers]得到最后的异常点
        """
        if not isinstance(ts, pd.Series):
            raise ValueError("'ts' should be pandas.Series, you got: {}".format(type(ts)))

        self.data = ts
        y = ts.values
        t = np.arange(len(ts))

        y_adj, seasonal_flag, seasonality = self.seasonal_adjust(ts)
        if seasonal_flag:
            y = y_adj

        self.smoother.fit(t, y)
        smoothed_data = self.smoother.predict(t)

        diff = y - smoothed_data
        outliers, upper_limit, lower_limit = box_method(diff)

        self.outliers = outliers
        self.smoothed_data = pd.Series(smoothed_data, index=ts.index) + seasonality
        self.upper_limit, self.lower_limit = upper_limit, lower_limit

        return outliers

    def seasonal_adjust(self, ts: pd.Series):
        """
        seasonal adjustment
        :param ts:
        :return:
        """
        from statsmodels.tsa.api import STL

        seasonal_flag = False
        seasonality = [0 for _ in range(len(ts))]
        y = ts.values

        if self.seasadj:
            try:
                fit = STL(ts, robust=True).fit()
                rem = fit.resid
                detrend = ts - fit.trend
                strength = 1 - np.var(rem) / np.var(detrend)

                if strength >= 0.6:
                    y = (fit.trend + fit.resid).values
                    seasonality = fit.seasonal
                    seasonal_flag = True
            except Exception as e:
                print(e)

        return y, seasonal_flag, seasonality

    def plot(self, verbose=False):
        """
        :param verbose: True or False，是否画出经过supersmoother平滑后的曲线以及上下界
        :return:
        """
        import matplotlib.pyplot as plt
        plt.rc('figure', figsize=(12, 5))

        ts, outliers = self.data, self.outliers

        plt.plot(ts, label='data')
        plt.plot(ts[outliers], 'ro', label='outliers')

        if verbose:
            smoothed_data = self.smoothed_data
            plt.plot(smoothed_data, label='smoothed data')

            diff = ts - smoothed_data
            plt.plot(diff, label='diff')

            upper_limit, lower_limit = self.upper_limit, self.lower_limit
            y_upper = np.ones(len(ts)) * upper_limit
            y_lower = np.ones(len(ts)) * lower_limit
            plt.plot(ts.index, y_upper, 'r--', label='upper limit')
            plt.plot(ts.index, y_lower, 'r--', label='lower limit')

        plt.legend()
        plt.show()


def box_method(diff, c0=(0.25, 0.75), c1=3):
    """
    box plot方法去做异常检测
    :param diff: data to detect anomaly
    :param c0: 25%和75% 分位数
    :param c1:
    :return:
    """

    quantile_lower = np.quantile(diff, c0[0])
    quantile_upper = np.quantile(diff, c0[1])
    iqr = quantile_upper - quantile_lower

    lower_limit = quantile_lower - c1 * iqr
    upper_limit = quantile_upper + c1 * iqr

    outliers = (diff >= upper_limit) | (diff <= lower_limit)

    return outliers, upper_limit, lower_limit


if __name__ == '__main__':
    # simulate time series data
    np.random.seed(2020)
    N = 365
    ts = pd.Series(
        (np.arange(N) // 40 + np.arange(N) % 21 + np.random.randn(N)),
        index=pd.DatetimeIndex(pd.date_range('2019-1-25', periods=N, freq='D'))
    )

    # anomaly detection
    ad = SmootherAD(seasadj=True)
    outliers = ad.detect(ts)
    ad.plot()
