#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from analysis.plot import plotDotData, displayPlot, plotFront, plotLineData, savePlot
from expression.tools import rmtocm
import logging
import json
logger = logging.getLogger('global')

class Plotter:
    def __init__(self):
        self._plots = []

    def plotDataPoints(self, X):
        p= plotDotData(X, labelx="X", labely="Y", title="Data Points")
        self._plots.append(p)

    def displayPlots(self, filename, title):
        displayPlot(self._plots, filename, title)

    def savePlots(self, filename, title):
        savePlot(self._plots, filename, title)

    def addPlot(self, p):
        self._plots.append(p)

class Convergence(Plotter):
    """
        Utility class to parse and convert convergence statistics.
    """
    def __init__(self, convergencestats):
        super().__init__()
        self._convergencestats = convergencestats
        self._runs = len(self._convergencestats)

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
        self.addPlot(p)

    def plotComplexity(self):
        """
            Plot complexity values over the generations
        """
        converted = []
        generations = 0
        cvalues = []
        runs = len(self._convergencestats)
        for i, run in enumerate(self._convergencestats):
            if i == 0:
                generations = len(run[0])
            cvalues += [gen['complexity'] for gen in run]
        converted = rmtocm(cvalues)
        logger.debug("Have {} runs and {} generations".format(runs, generations))
        p = plotDotData(converted, labelx="Generation", labely="Complexity", title="Complexity")
        self.addPlot(p)


    def plotOperators(self):
        """
            Plot effective (i.e. rendering a fitter individual) modifications made by mutations and crossover
        """
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
        self.addPlot(p)

    def plotPareto(self):
        """
            Plot fitness values against complexity in a front
        """
        run = self._convergencestats[-1]
        fitnessvalues = run[-1]['fitness']
        complexity = run[-1]['complexity']
        logger.debug("Plotting fitness {} against complexity {}".format(fitnessvalues, complexity))
        p = plotFront(X=fitnessvalues, Y=complexity, labelx="Fitness", labely="complexity", title="Front")
        self.addPlot(p)

    def saveData(self, filename, outputfolder=None):
        logger.info("writing to output {} in folder {}".format(filename, outputfolder))
        outputfolder = outputfolder or ""
        with open(outputfolder+filename, 'w') as filename:
            json.dump(self._convergencestats, filename, indent=4)


class SummarizedResults(Plotter):
    def __init__(self, results):
        super().__init__()
        self._results = results

    def plotFitness(self):
        print(results)
        # fitness = []
        # p = plotDotData(fitness, labelx="Generation", labely="Fitness", title="Fitness")


    def plotComplexity(self):
        pass

    def saveData(self, filename, outputfolder=None):
        logger.info("writing to output {} in folder {}".format(filename, outputfolder))
        outputfolder = outputfolder or ""
        with open(outputfolder+filename, 'w') as filename:
            json.dump(self._results, filename, indent=4)
