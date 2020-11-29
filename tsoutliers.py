import numpy as np
import pandas as pd
from statsmodels.tsa.api import STL

from supersmoother.supersmoother import SuperSmoother


class SmootherAD:

    def __init__(self, seasadj=False):
        self.seasadj = seasadj
        self.smoother = SuperSmoother()

        self.smoothed_data = None

    def detect(self, ts: pd.Series, is_plot=False):
        """
        :param ts: time series to detect anomalies
        :param is_plot:
        :return:
        """
        if not isinstance(ts, pd.Series):
            raise ValueError("'ts' should be pandas.Series, you got: {}".format(type(ts)))

        y = ts.values
        t = np.arange(len(ts))

        y_adj, seasonal_flag, seasonality = self.seasonal_adjust(ts)
        if seasonal_flag:
            y = y_adj

        self.smoother.fit(t, y)
        smoothed_data = self.smoother.predict(t)

        outliers, diff, upper_limit, lower_limit = self.box_method(y, smoothed_data)

        self.smoothed_data = pd.Series(smoothed_data, index=ts.index) + seasonality

        if is_plot:
            self.plot(ts, self.smoothed_data, outliers, upper_limit, lower_limit)

        return outliers

    def seasonal_adjust(self, ts: pd.Series):
        """
        seasonal adjustment
        :param ts:
        :return:
        """
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

    @staticmethod
    def box_method(y, smoothed_data, c0=(0.25, 0.75), c1=3):
        """
        using box plot method to detect final anomalies
        :param y: real data
        :param smoothed_data: data after super smooth
        :param c0: quantile
        :param c1:
        :return:
        """
        diff = y - smoothed_data

        quantile_lower = np.quantile(diff, c0[0])
        quantile_upper = np.quantile(diff, c0[1])
        iqr = quantile_upper - quantile_lower

        lower_limit = quantile_lower - c1 * iqr
        upper_limit = quantile_upper + c1 * iqr

        outliers = (diff >= upper_limit) | (diff <= lower_limit)

        return outliers, diff, upper_limit, lower_limit

    @staticmethod
    def plot(ts: pd.Series, smoothed_data: pd.Series, outliers, upper_limit, lower_limit):
        """
        plot the image
        :param ts:
        :param smoothed_data:
        :param outliers:
        :param upper_limit:
        :param lower_limit:
        :return:
        """
        import matplotlib.pyplot as plt
        plt.rc('figure', figsize=(12, 5))

        plt.plot(ts, label='data')
        plt.plot(smoothed_data, label='smoothed data')
        plt.plot(ts[outliers], 'ro', label='outliers')

        y_upper = np.ones(len(ts)) * upper_limit
        y_lower = np.ones(len(ts)) * lower_limit
        plt.plot(ts.index, y_upper, 'r--', label='upper limit')
        plt.plot(ts.index, y_lower, 'r--', label='lower limit')

        plt.legend()
        plt.show()


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
    outliers = ad.detect(ts, is_plot=True)
