#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from analysis.plot import plotDotData, displayPlot, plotFront, plotLineData
from expression.tools import rmtocm
import logging
logger = logging.getLogger('global')

class Convergence:
    def __init__(self, convergencestats):
        self._convergencestats = convergencestats
        self._runs = len(self._convergencestats)
        self._plots = []

    def plotFitness(self):
        """
            Plot fitness values over the generations
        """
        converted = []
        generations = 0
        fitnessvalues = []
        runs = len(self._convergencestats)
        logger.debug("Have {} runs and {} generations".format(runs, generations))
        for i, run in enumerate(self._convergencestats):
            if i == 0:
                generations = len(run[0])
            fitnessvalues += [gen['fitness'] for gen in run]
        converted = rmtocm(fitnessvalues)
        p = plotDotData(converted, labelx="Generation", labely="Fitness", title="Fitness")
        self._plots.append(p)

    def plotComplexity(self):
        """
            Plot complexity values over the generations
        """
        converted = []
        generations = 0
        cvalues = []
        runs = len(self._convergencestats)
        logger.info("Have {} runs and {} generations".format(runs, generations))
        for i, run in enumerate(self._convergencestats):
            if i == 0:
                generations = len(run[0])
            cvalues += [gen['complexity'] for gen in run]
        converted = rmtocm(cvalues)
        p = plotDotData(converted, labelx="Generation", labely="Complexity", title="Complexity")
        self._plots.append(p)

    def plotOperators(self):
        # Plot replacements, crossovers and mutations
        converted = []
        generations = 0
        cvalues = [[],[]]
        runs = len(self._convergencestats)
        logger.debug("Have {} runs and {} generations".format(runs, generations))
        for i, run in enumerate(self._convergencestats):
            if i == 0:
                generations = len(run[0])
#            cvalues[0] += [gen['replacements'] for gen in run]
            cvalues[0] += [gen['mutations'] for gen in run]
            cvalues[1] += [gen['crossovers'] for gen in run]
        p = plotLineData(cvalues, labelx="Generation", labely="Successful operations", title="Modifications", legend=["Mutations","Crossovers"])
        self._plots.append(p)

    def plotPareto(self):
        """
            Plot fitness values against complexity in a front
        """
        run = self._convergencestats[-1]
        fitnessvalues = run[-1]['fitness']
        complexity = run[-1]['complexity']
        logger.debug("Plotting fitness {} against complexity {}".format(fitnessvalues, complexity))
        p = plotFront(X=fitnessvalues, Y=complexity, labelx="Fitness", labely="complexity", title="Front")
        self._plots.append(p)

    def displayPlots(self, filename):
        displayPlot(self._plots, filename)
