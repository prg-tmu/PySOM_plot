import os
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from statistics import geometric_mean, variance
from pprint import pprint


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


class PySOMPlotGC:
    def __init__(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            self.path = path
        else:
            raise Exception(path + " is not directory")

        self.data = {}
        self.script_dir = os.path.dirname(__file__)
        self.path_files = sorted(pathlib.Path(self.path).glob("*.txt"))
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

        # pprint(self.data)

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

    def plot(self):
        benchmarks = self.data.keys()
        executables = ["interp", "threaded", "tracing"]

        minor = {}
        major = {}

        for i, exe in enumerate(self.data):
            minor[exe] = {}
            major[exe] = {}

            for name in self.data[exe]:
                minor[exe][name] = {}
                major[exe][name] = {}

                duration_minors = []
                duration_majors = []

                for inv in self.data[exe][name]:
                    gc_info = self.data[exe][name][inv]
                    duration_minors.append(gc_info.duration_minor)
                    duration_majors.append(gc_info.duration_major)

                n = 100
                duration_minor_gmean = geometric_mean(duration_minors) / n
                duration_minor_var = variance(duration_minors) / n
                duration_major_gmean = geometric_mean(duration_majors) / n
                duration_major_var = variance(duration_majors) / n

                minor[exe][name] = duration_minor_gmean
                major[exe][name] = duration_major_gmean

        df_minor = pd.DataFrame(minor)
        df_major = pd.DataFrame(major)

        df_minor["threaded"] = df_minor["threaded"] / df_minor["interp"]
        df_minor["tracing"] = df_minor["tracing"] / df_minor["interp"]
        df_minor["interp"] = df_minor["interp"] / df_minor["interp"]

        df_major["threaded"] = df_major["threaded"] / df_major["interp"]
        df_major["tracing"] = df_major["tracing"] / df_major["interp"]
        df_major["interp"] = df_major["interp"] / df_major["interp"]

        del df_minor["interp"]
        del df_major["interp"]

        df_minor.plot.bar(
            xlabel="Benchmarks",
            ylabel="GC minor standarlized\nto interpreter (lower is better)",
        )
        plt.savefig('gc_minor.pdf', bbox_inches='tight')
        df_major.plot.bar(
            xlabel="Benchmarks",
            ylabel="GC major standarlized\nto interpreter (lower is better)",
        )
        plt.savefig('gc_major.pdf', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    pysomplot_gc = PySOMPlotGC(path)
    pysomplot_gc.plot()
