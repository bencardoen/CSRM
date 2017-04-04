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
import random
from expression.constants import Constants
logger = logging.getLogger('global')

plotwidth = 100
plotheigth = 150


def groupData(data, scale=1):
    """
    Given a linear array of data values, sort them by value, count each instance.

    Return a dictionary where each value points to the relative frequency multiplied by scale.
    E.g. [1,2,2], scale=2 -> {1:2/3, 2:4/3}
    """
    dlen = len(data)
    collected = {}
    for x in data:
        if x in collected:
            collected[x] += 1
        else:
            collected[x] = 1
    for k, v in collected.items():
        collected[k] = (v / dlen) * scale
    return collected


def plotDotData(data, mean=None, std=None, var=None, generationstep=1, labelx=None, labely=None, title=None, cool=False, xcategorical=False, ycategorical=False, groupsimilar=False, scalesimilar=1):
    """
    Plot data values over generations.
    """
    logger.debug("Got {} to plot".format(data))
    labelx = labelx or "X"
    labely = labely or "Y"
    dlen = len(data)
    xranges = [str(x) for x in range(len(data[0]))] if xcategorical else None
    yranges = [str(x) for x in range(1, max( [ max(v) for v in data]) +1)] if ycategorical else None
    p = figure(title=title or "title", x_axis_label=labelx, y_axis_label=labely, x_range=xranges, y_range=yranges)
    x = [d+1 for d in range(len(data[0]))]
    dlen = len(data)
    colors = ["navy" for _ in range(dlen)]
    # todo constants
    size = Constants.PLOT_SIZE_DEFAULT
    alpha=Constants.PLOT_ALPHA_DEFAULT
    if cool:
        size = Constants.PLOT_SIZE_COOL
        if dlen <= 256:
            colors = viridis(dlen)
        else:
            colors = viridis(256)
            colors += [colors[-1] for _ in range(dlen-256)]
    if groupsimilar:
        size = Constants.PLOT_SIZE_DEPTH
        alpha = Constants.PLOT_ALPHA_DEPTH
    for i,d in enumerate(data):
        p.circle(x, d, size=size, color=colors[dlen-i-1], alpha=alpha)
    return p


def plotLineData(data, mean=None, std=None, var=None, generationstep=1, labelx=None, labely=None, title=None, legend=None, xcategorical=False, ycategorical=False, dot=False):
    """
    Plot data values over generations.
    """
    dlen = len(data)
    colors = None
    if dlen <= 256:
        colors = inferno(dlen+1)[:-1]
    else:
        colors = inferno(256)
        colors += [colors[-1] for _ in range(dlen-256)]
    labelx = labelx or "X"
    labely = labely or "Y"
    #xranges = [str(x) for x in range(0, dlen)] if xcategorical else None
    xranges = None
    yranges = [str(x) for x in range(1, max( [ max(v) for v in data]) +1)] if ycategorical else None
    if legend is None:
        legend = [d for d in range(len(data))]
    p = figure(title=title or "title", x_axis_label=labelx, y_axis_label=labely, x_range=xranges, y_range=yranges)
    for i,d in enumerate(data):
        x = [d for d in range(len(data[0]))]
        if dot:
            p.circle(x, d, line_width=1, line_color=colors[i], fill_color=colors[i], line_alpha=0.5, legend=legend[i])
        else:
            p.line(x, d, line_width=2, line_color=colors[i], line_alpha=1, legend=legend[i])
    return p


def plotFront(X, Y, labelx=None, labely=None, title=None):
    p = figure(title=title or "title", x_axis_label=labelx or "X", y_axis_label=labely or "Y")
    p.circle(X, Y, size=2, color="red", alpha=0.3)
    return p


def displayPlot(plots, filename, title, square=False):
    output_file("{}.html".format(filename or "index"), title=title or "title")
    gplots = plots
    if square:
        gplots = []
        if len(plots) % 2:
            plots.append(None)
        gplots = [[plots[2*d], plots[2*d+1]] for d in range(len(plots)//2)]
    p = gridplot(*gplots, ncols=3, plot_width=500, plot_height=500)
    show(p)


def savePlot(plots, filename, title, square=False):
    output_file("{}.html".format(filename or "index"), title=title or "title")
    gplots = plots
    if square:
        gplots = []
        if len(plots) % 2:
            plots.append(None)
        gplots = [[plots[2*d], plots[2*d+1]] for d in range(len(plots)//2)]
    p = gridplot(*gplots, ncols = 4)
    save(p)
