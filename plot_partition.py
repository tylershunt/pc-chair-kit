#!/usr/bin/env python3

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

"""
take full list of splits and plot thier socres for comparison
"""

parts = [p for p in csv.DictReader(sys.stdin)]

scores = [float(p["score"]) for p in parts]
scores.sort()

matplotlib.pyplot.scatter(range(len(scores)), scores, s=1)
matplotlib.pyplot.savefig("splits_graph.png")
