import matplotlib.style as style
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib

import util

class Base(object):
    def __init__(self, path, standardize, suffix):
        self.path = util.path(path, is_dir=True)
        self.standardize = standardize
        self.suffix = suffix
        self.path_files = sorted(pathlib.Path(self.path).glob("*" + suffix))

        self.max_invocation = -1

        sns.set_style("darkgrid")
        sns.set_context("paper")

    def _savefig(self, name, *args, **kwargs):
        if not os.path.isdir('output'):
            os.mkdir('output')

        if 'output' not in name:
            name = "output/" + name

        plt.tight_layout()
        plt.savefig(name, **kwargs)
