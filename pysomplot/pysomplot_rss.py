import os
import re
import sys
import pathlib

import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pprint import pprint
from statistics import geometric_mean, variance

from base import Base

class PySOMPlotRss(Base):
    def __init__(self, path, standardize, suffix=".txt"):
        super().__init__(path, standardize, suffix)

        self.data = {}

    def _setup_data(self, name, exe):
        if exe not in self.data:
            self.data[exe] = {}
        if name not in self.data[exe]:
            self.data[exe][name] = {}

    def parse_files(self):
        for p in self.path_files:
            fpath = os.path.join(self.path, p.name)

            name = p.name
            basename = name.removesuffix(".txt")
            name, exe, inv = basename.split("_")
            self._setup_data(name, exe)

            with open(fpath, "r") as f:
                while line := f.readline():
                    pattern = "'RSS:(\d+) KB'"
                    r = re.match(pattern, line)
                    if r:
                        inv = int(inv)
                        rss =  float(r.group(1)) / 1024
                        self.data[exe][name][inv] = rss

    def plot(self):
        self.parse_files()

        data_gmean = {}
        data_var = {}

        for exe in self.data:
            for name in self.data[exe]:
                values = self.data[exe][name].values()
                gmean = geometric_mean(values)
                var = variance(values)
                if exe not in data_gmean:
                    data_gmean[exe] = {}

                if exe not in data_var:
                    data_var[exe] = {}

                if name not in data_gmean[exe]:
                    data_gmean[exe][name] = {}

                if name not in data_var[exe]:
                    data_var[exe][name] = {}

                data_gmean[exe][name] = gmean
                data_var[exe][name] = var

        df = pd.DataFrame(data_gmean)
        ax = df.plot.bar(
            xlabel="Benchmarks",
            ylabel="RSS (MB, lower is better)",
            yerr=pd.DataFrame(data_var),
            capsize=2,
            figsize=(10, 6)
        )
        self._savefig("rss.pdf")
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="A plotting script for PySOM")
    parser.add_argument('dirname')
    parser.add_argument(
        "-s", "--standardize", action="store_true", help="Standardized to interpreter"
    )
    args = parser.parse_args()
    path = args.dirname
    pysomplot_rss = PySOMPlotRss(path, args.standardize)
    pysomplot_rss.plot()
