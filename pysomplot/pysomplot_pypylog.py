import argparse
import sys
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

from base import Base

class PySOMPlotPyPyLog(Base):
    def __init__(self, path, standardize, suffix=".log"):
        super().__init__(path, standardize, suffix)

        self.data = {}

        sns.set_style("darkgrid")
        sns.set_context("paper")

    def _setup_data(self, name, exe):
        if exe not in self.data:
            self.data[exe] = {}
        if name not in self.data[exe]:
            self.data[exe][name] = {}

    def parse_files(self):
        for p in self.path_files:
            fpath = os.path.join(self.path, p.name)

            name = p.name
            basename = name.removesuffix(self.suffix)
            name, exe, inv = basename.split("_")
            self._setup_data(name, exe)

            with open(fpath, "r") as f:
                while line := f.readline():
                    pat = "\w+:\s\s*\t\d+\t(\d\.\d+)"
                    r = re.match(pat, line)
                    if r:
                        inv = int(inv)
                        if self.max_invocation < inv:
                            self.max_invocation = inv

                        time = float(r.group(1))
                        if inv in self.data[exe][name]:
                            self.data[exe][name][inv] += time
                        else:
                            self.data[exe][name][inv] = time


    def plot(self):
        import pprint
        import statistics as st

        self.parse_files()

        result = {}

        for exe in self.data:
            for name in self.data[exe]:
                for i in range(self.max_invocation):
                    if exe not in result:
                        result[exe] = {}

                    if name not in result[exe]:
                        result[exe][name] = []

                    result[exe][name].append(self.data[exe][name][i])


        medians = {}
        variances = {}
        for exe in result:
            medians[exe] = {}
            variances[exe] = {}
            for name in result[exe]:
                medians[exe][name] = {}
                variances[exe][name] = {}
                medians[exe][name] = st.median(result[exe][name]) * 1e3 # second to millisecond
                variances[exe][name] = st.variance(result[exe][name]) * 1e3

        df_bytesize = pd.read_csv("data/benchmark_bytesize.csv")
        df_bytesize["program"] = df_bytesize["program"].str.lower()
        df_bytesize = df_bytesize.set_index("program")
        df_bytesize = df_bytesize.drop("json")

        df_med = pd.DataFrame(medians)
        df_var = pd.DataFrame(variances)

        df = df_bytesize.join(df_med)

        x = df["size"]
        y_threaded = df["threaded"]
        y_tracing = df["tracing"]
        yerr_threaded = df_var["threaded"]
        yerr_tracing = df_var["tracing"]

        fig, ax = plt.subplots()
        ax.scatter(x, y_threaded, c='tab:blue', s=100)
        ax.scatter(x, y_tracing, c='tab:red', s=100)

        x1, x2, y1, y2 = -100, 1750, -25, 1100
        axins = ax.inset_axes(
            [0.42, 0.42, 0.58, 0.58],
            xlim=(x1, x2), ylim=(y1, y2),
        )

        mod = LinearRegression()

        df_x = pd.DataFrame(x).drop("pagerank")
        df_y = pd.DataFrame(y_threaded).drop("pagerank")
        mod_lin = mod.fit(df_x, df_y)
        y_lin_fit = mod_lin.predict(df_x)
        r2_lin = mod.score(df_x, df_y)

        axins.scatter(x, y_threaded, c='tab:blue', s=100)
        axins.scatter(x, y_tracing, c='tab:red', s=100)

        axins.plot(df_x, y_lin_fit, color='tab:blue', linewidth=2)
        axins.text(1250, 50, '$ y = $' + str(round(mod.coef_[0][0], 4)) + " x + " + str(round(mod.intercept_[0], 2)), fontsize=12)
        axins.text(1250, 100, '$ R^{2} $=' + str(round(r2_lin, 4)), fontsize=12)

        df_x = pd.DataFrame(x).drop("pagerank")
        df_y = pd.DataFrame(y_tracing).drop("pagerank")
        mod_lin = mod.fit(df_x, df_y)
        y_lin_fit = mod_lin.predict(df_x)
        r2_lin = mod.score(df_x, df_y)

        axins.plot(df_x, y_lin_fit, color='tab:red', linewidth=2)
        axins.text(100, 300, '$ y = $' + str(round(mod.coef_[0][0], 4)) + " x + " + str(round(mod.intercept_[0], 2)), fontsize=12)
        axins.text(100, 350, '$ R^{2} $=' + str(round(r2_lin, 4)), fontsize=12)

        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A plotting script for PySOM")
    parser.add_argument('dirname')
    parser.add_argument(
        "-s", "--standardize", action="store_true", help="Standardized to interpreter"
    )
    args = parser.parse_args()
    path = args.dirname
    pysomplot_pypylog = PySOMPlotPyPyLog(path, args.standardize)
    pysomplot_pypylog.plot()
