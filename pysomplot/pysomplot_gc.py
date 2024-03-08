import os
import sys
import pathlib

import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pprint import pprint
from statistics import geometric_mean, variance, median

from base import Base
import util

class GcInfo:
    def __init__(self, minor, collect_step, collect, duration_minor, duration_major):
        self.minor = minor
        self.collect_step = collect_step
        self.collect = collect
        self.duration_minor = duration_minor
        self.duration_major = duration_major

    def __repr__(self):
        return "GcInfo(%d, %d, %d, %f, %f)" % (
            self.minor,
            self.collect_step,
            self.collect,
            self.duration_minor,
            self.duration_major,
        )


class PySOMPlotGC(Base):
    def __init__(self, path, standardize, suffix=".txt"):
        super().__init__(path, standardize, suffix)

        self.data = {}
        self.script_dir = os.path.dirname(__file__)
        for p in self.path_files:
            name = p.name
            basename = name.removesuffix(".txt")
            name, exe, inv = basename.split("_")
            gc_info = self._read_file(os.path.join(self.path, p.name))
            if gc_info:
                if exe not in self.data:
                    self.data[exe] = {}

                if name not in self.data[exe]:
                    self.data[exe][name] = {}

                if not self.data[exe][name]:
                    self.data[exe][name] = {}

                self.data[exe][name][int(inv)] = gc_info


    def _read_file(self, fpath):
        with open(fpath, "r") as f:
            while line := f.readline():
                line = line.strip()
                if "gc-minor:" in line:
                    num_gc_minor = int(line.split(":")[1].strip())

                if "gc-collect-step:" in line:
                    num_gc_collect_step = int(line.split(":")[1].strip())

                if "gc-collect" in line:
                    num_gc_collect = int(line.split(":")[1].strip())

                if "gc-duration-minor" in line:
                    num_gc_duration_minor = float(
                        line.split(":")[1].strip().split(" ")[0]
                    )

                if "gc-duration-major" in line:
                    num_gc_duration_major = float(
                        line.split(":")[1].strip().split(" ")[0]
                    )

            return GcInfo(
                num_gc_minor,
                num_gc_collect_step,
                num_gc_collect,
                num_gc_duration_minor,
                num_gc_duration_major,
            )

    def _standardize(self, df, exe, exe_base):
        df[exe] = df[exe] / df[exe_base]
        return df

    def plot(self):
        benchmarks = self.data.keys()
        executables = ["interp", "threaded", "tracing"]

        minor = {}
        minor_var = {}
        major = {}
        major_var = {}

        for i, exe in enumerate(self.data):
            minor[exe] = {}
            major[exe] = {}
            minor_var[exe] = {}
            major_var[exe] = {}

            for name in self.data[exe]:
                minor[exe][name] = {}
                major[exe][name] = {}
                minor_var[exe][name] = {}
                major_var[exe][name] = {}

                duration_minors = []
                duration_majors = []

                for inv in self.data[exe][name]:
                    gc_info = self.data[exe][name][inv]
                    duration_minors.append(gc_info.duration_minor)
                    duration_majors.append(gc_info.duration_major)

                n = 100
                duration_minor_gmean = median(duration_minors) / n  # geometric_mean(duration_minors) / n
                duration_minor_var = variance(duration_minors) / n
                duration_major_gmean = median(duration_majors) / n  # geometric_mean(duration_majors) / n
                duration_major_var = variance(duration_majors) / n

                minor[exe][name] = duration_minor_gmean
                major[exe][name] = duration_major_gmean
                minor_var[exe][name] = duration_minor_var
                major_var[exe][name] = duration_major_var

        df_minor = pd.DataFrame(minor)
        df_minor_var = pd.DataFrame(minor_var)
        df_major = pd.DataFrame(major)
        df_major_var = pd.DataFrame(major_var)

        if self.standardize:
            df_minor["threaded"] = df_minor["threaded"] / df_minor["interp"]
            df_minor["tracing"] = df_minor["tracing"] / df_minor["interp"]
            df_minor["interp"] = df_minor["interp"] / df_minor["interp"]

            df_minor_var["threaded"] = df_minor_var["threaded"] / df_minor_var["interp"]
            df_minor_var["tracing"] = df_minor_var["tracing"] / df_minor_var["interp"]
            df_minor_var["interp"] = df_minor_var["interp"] / df_minor_var["interp"]

            df_major["threaded"] = df_major["threaded"] / df_major["interp"]
            df_major["tracing"] = df_major["tracing"] / df_major["interp"]
            df_major["interp"] = df_major["interp"] / df_major["interp"]

            df_major_var["threaded"] = df_major_var["threaded"] / df_major_var["interp"]
            df_major_var["tracing"] = df_major_var["tracing"] / df_major_var["interp"]
            df_major_var["interp"] = df_major_var["interp"] / df_major_var["interp"]

            del df_minor["interp"]
            del df_minor_var["interp"]
            del df_major["interp"]
            del df_major_var["interp"]

            ax = df_minor.plot.bar(
                xlabel="Benchmarks",
                ylabel="GC minor time\nstandardized to interpreter (lower is better)",
                yerr=df_minor_var,
                figsize=(10, 6)
            )
            ax.axhline(y=1, color='green', linewidth=2)
            self._savefig('gc_minor_standardized.pdf', bbox_inches='tight')

            ax = df_major.plot.bar(
                xlabel="Benchmarks",
                ylabel="GC major time\nstandardizedto interpreter (lower is better)",
                yerr=df_major_var,
                figsize=(10, 6)
            )
            ax.axhline(y=1, color='green', linewidth=2)
            self._savefig('gc_major_standardized.pdf', bbox_inches='tight')

        else:
            ax = df_minor.plot.bar(
                xlabel="Benchmarks",
                ylabel="GC minor (us)",
                yerr=df_minor_var,
                figsize=(10, 6)
            )
            self._savefig('gc_minor.pdf', bbox_inches='tight')

            ax = df_major.plot.bar(
                xlabel="Benchmarks",
                ylabel="GC major (us)",
                yerr=df_major_var,
                figsize=(10, 6)
            )
            self._savefig('gc_major.pdf', bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="A plotting script for PySOM")
    parser.add_argument('dirname')
    parser.add_argument(
        "-s", "--standardize", action="store_true", help="Standardized to interpreter"
    )
    args = parser.parse_args()
    path = args.dirname
    pysomplot_gc = PySOMPlotGC(path, args.standardize)
    pysomplot_gc.plot()
