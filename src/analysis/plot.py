#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
from bokeh.plotting import figure, output_file, show, gridplot, save
from bokeh.palettes import magma, inferno, viridis
import logging
import math
logger = logging.getLogger('global')
import random

plotwidth = 200
plotheigth = 200

def plotDotData(data, mean=None, std=None, var=None, generationstep=1, labelx=None, labely=None, title=None, cool=False):
    """
        Plot data values over generations.
    """
    logger.debug("Got {} to plot".format(data))
    labelx = labelx or "X"
    labely = labely or "Y"
    p = figure(title=title or "title", x_axis_label=labelx, y_axis_label=labely)
    x = [d for d in range(len(data[0]))]
    dlen = len(data)
    ratio = 10/dlen
    colors = ["navy" for _ in range(dlen)]
    if cool:
        diff = 0
        if dlen <= 256:
            colors = viridis(dlen)
        else:
            colors = viridis(256)
            colors += [colors[-1] for _ in range(dlen-256)]
    for i,d in enumerate(data):
        p.circle(x, d, size=4 if cool else 1, color=colors[dlen-i-1], alpha=0.5)
    return p

def plotLineData(data,  mean=None, std=None, var=None, generationstep=1, labelx=None, labely=None, title=None, legend=None):
    """
        Plot data values over generations.
    """
    colors = ["blue", "green", "red"]
    logger.debug("Got {} to plot".format(data))
    labelx = labelx or "X"
    labely = labely or "Y"
    if legend is None:
        legend = [d for d in range(len(data))]
    p = figure(title=title or "title", x_axis_label=labelx, y_axis_label=labely)
    for i,d in enumerate(data):
        x = [d for d in range(len(data[0]))]
        p.circle(x, d, line_width=1, line_color=colors[i], line_alpha=0.5, legend=legend[i])
    return p

def plotFront(X, Y, labelx=None, labely=None, title=None):
    p = figure(title=title or "title", x_axis_label=labelx or "X", y_axis_label=labely or "Y")
    p.circle(X, Y, size=2, color="red", alpha=0.3)
    return p

def displayPlot(plots, filename, title):
    output_file("{}.html".format(filename or "index"), title=title or "title")
    gplots = []
    if len(plots) % 2:
        plots.append(None)
    gplots = [[plots[2*d], plots[2*d+1]] for d in range(len(plots)//2)]
    p = gridplot(*gplots)
    show(p)

def savePlot(plots, filename, title):
    output_file("{}.html".format(filename or "index"), title=title or "title")
    gplots = []
    if len(plots) % 2:
        plots.append(None)
    gplots = [[plots[2*d], plots[2*d+1]] for d in range(len(plots)//2)]
    p = gridplot(*gplots)
    save(p)

if __name__ == "__main__":
    rng = random.Random()
    fitness = [[random.random() for d in range(100)] for d in range(10)]
    p = plotFitness(fitness)
    show(p)
