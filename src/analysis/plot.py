#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
from bokeh.plotting import figure, output_file, show
import logging
import math
logger = logging.getLogger('global')
import random

def plotDotData(data, mean=None, std=None, var=None, generationstep=1, labelx=None, labely=None, title=None):
    """
        Plot data values over generations.
    """
    logger.debug("Got {} to plot".format(data))
    output_file("{}.html".format(title or "title"))
    labelx = labelx or "X"
    labely = labely or "Y"
    p = figure(title=title or "title", x_axis_label=labelx, y_axis_label=labely)
    x = [d for d in range(len(data[0]))]
    for i,d in enumerate(data):
        logger.debug("Plotting {}".format(d))
        p.circle(x, d, size=1, color="navy", alpha=0.5)
    return p

def plotFront(X, Y, labelx=None, labely=None, title=None):
    output_file("{}.html".format(title or "title"))
    p = figure(title=title or "title", x_axis_label=labelx or "X", y_axis_label=labely or "Y")
    p.circle(X, Y, size=2, color="red", alpha=0.3)
    return p

def displayPlot(p):
    show(p)

if __name__ == "__main__":
    rng = random.Random()
    fitness = [[random.random() for d in range(100)] for d in range(10)]
    p = plotFitness(fitness)
    show(p)
