import os
import sys
import math
import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("ggplot")

from scipy.stats.mstats import gmean
from pprint import pprint
from numpy.lib import type_check


class PySOMPlot:
    def __init__(self, filename):
        self.filename = filename
        self.basename = os.path.basename(get_name(self.filename))

        self.results_teir1 = None
        self.results_interp = None
        self.resulsts_tier2 = None

        self.benchmarks = []
        self.executors = []

        self.results = {}

        try:
            f = open(self.filename, "r")
        except IOError:
            raise Exception("not found", filename)


        while True:
            line = f.readline().rstrip()
            if line.startswith("#"):
                continue
            if len(line) == 0:
                break

            line = line.split("\t")
            try:
                invocation, iteration, elapsed, benchmark, executor = (
                    float(line[0]),
                    float(line[1]),
                    float(line[2]),
                    line[5],
                    line[6],
                )
            except Exception:
                continue

            if executor not in self.executors:
                self.executors.append(executor)

            if benchmark not in self.benchmarks:
                self.benchmarks.append(benchmark)

            if executor not in self.results:
                self.results[executor] = {}

            if benchmark not in self.results[executor]:
                self.results[executor][benchmark] = {}

            if not self.results[executor][benchmark]:
                self.results[executor][benchmark] = [{invocation: [elapsed]}]
            else:
                is_existed = False
                for d in self.results[executor][benchmark]:
                    if invocation in d:
                        d[invocation].append(elapsed)
                        is_existed = True
                        break

                if not is_existed:
                    self.results[executor][benchmark].append({invocation: [elapsed]})



        self.benchmarks.sort()
        self.executors.sort()

    def plot_boxplots_invks(self):
        print(self.results)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        raise Exception("argument is not specified")
    pysom_plot = PySOMPlot(filename)
    pysom_plot.plot_boxplots_invks()
