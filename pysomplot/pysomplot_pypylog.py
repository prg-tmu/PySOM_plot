import argparse
import sys
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from base import Base

class PySOMPlotPyPyLog(Base):
    def __init__(self, path, standardize, suffix=".log"):
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
                medians[exe][name] = st.median(result[exe][name])
                variances[exe][name] = st.variance(result[exe][name])

        df_med = pd.DataFrame(medians)
        df_var = pd.DataFrame(variances)
        new_df_med = df_med["threaded"] / df_med["tracing"]
        new_df_var = df_var["threaded"] / df_var["tracing"]
        new_df_med.plot.bar()
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
