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
from expression.tools import copyObject
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

    def getPlots(self):
        return self._plots


class Convergence(Plotter):
    """
    Utility class to parse and convert convergence statistics.
    """

    def __init__(self, convergencestats):
        super().__init__()
        self._convergencestats = convergencestats
        self._runs = len(self._convergencestats)

    def plotPopulationOverGenerations(self, keyword="fitness", cool=False, xcategorical=False, ycategorical=False, groupsimilar=False):
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
            fitnessvalues += [gen[keyword] for gen in run]
        converted = rmtocm(fitnessvalues)
        logger.debug("{} rows {} cols".format(len(converted), len(converted[0])) )
        ystr = "".join((c.upper() if i ==0 else c for i, c in enumerate(keyword)))
        p = plotDotData(converted, labelx="Generation", labely=ystr, title=ystr,cool=cool, xcategorical=xcategorical, ycategorical=ycategorical, groupsimilar=groupsimilar)
        self.addPlot(p)

    def plotFitness(self):
        self.plotPopulationOverGenerations(keyword='fitness')

    def plotDepth(self):
        self.plotPopulationOverGenerations(keyword='depth', cool=False, xcategorical=True, ycategorical=True, groupsimilar=True)

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
        p = plotLineData(cvalues, labelx="Generation", labely="Succes ratio of operator", title="Modifications", legend=["Mutations","Crossovers"])
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
        fitness = rmtocm([r['fitness'] for r in self._results])
        for i in range(1):
            p = plotDotData(fitness, labelx="Process", labely="Fitness", title="Fitness of the last generation calculated on the test entire set (sample + test).", xcategorical=True)
            self.addPlot(p)

    def plotPrediction(self):
        fitness = rmtocm([r['corr_fitness'] for r in self._results])
        logger.debug("{} rows {} cols".format(len(fitness), len(fitness[0])) )
        p = plotDotData(fitness, labelx="Process", labely="Correlation (less is better)", title="Correlation between fitness on sample data and test data per phase best value", cool=True, xcategorical=True)
        self.addPlot(p)


    def plotDifference(self):
        fitness = rmtocm([r['diff_fitness'] for r in self._results])
        logger.debug("{} rows {} cols".format(len(fitness), len(fitness[0])) )
        p = plotDotData(fitness, labelx="Process", labely="Difference", title="Difference between fitness on sample data and test data per phase best value", xcategorical=True)
        self.addPlot(p)

    def plotComplexity(self):
        logger.warning("Unused function")
        pass

    def saveData(self, filename, outputfolder=None):
        logger.info("writing to output {} in folder {}".format(filename, outputfolder))
        outputfolder = outputfolder or ""
        with open(outputfolder+filename, 'w') as filename:
            json.dump(self._results, filename, indent=4)
