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
import numpy
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
        self.plotAll()

    def plotPopulationOverGenerations(self, keyword="fitness", cool=False, xcategorical=False, ycategorical=False, groupsimilar=False):
        converted = []
        fitnessvalues = []
        for i, run in enumerate(self._convergencestats):
            fitnessvalues += [gen[keyword] for gen in run]
        converted = rmtocm(fitnessvalues)
        ystr = "".join((c.upper() if i ==0 else c for i, c in enumerate(keyword)))
        p = plotDotData(converted, labelx="Generation", labely=ystr, title=ystr,cool=cool, xcategorical=xcategorical, ycategorical=ycategorical, groupsimilar=groupsimilar)
        self.addPlot(p)

    def plotFitness(self):
        self.plotPopulationOverGenerations(keyword='fitness')

    def plotDepth(self):
        self.plotPopulationOverGenerations(keyword='depth', cool=False, xcategorical=False, ycategorical=True, groupsimilar=True)

    def plotFoldingGains(self):
        self.plotSeries(keys = ['foldingsavings'], labels=["Generation","Reduction % of nodes. (more is better)"], title="Constant Folding effectiveness", legend=["Constant folding."])

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
        self.plotSeries(keys = ['mutations','crossovers'], labels=["Generation","Succes ratio of operator"], title="Modfications", legend=["Mutations","Crossovers"])

    def plotOperatorsImpact(self):
        self.plotSeries(keys = ['mutate_gain','crossover_gain'], labels=["Generation","Mean gain of operator"], title="Impact of operator", legend=["Mutations","Crossovers"])

    def plotSeries(self, keys, labels, title, legend):
        cvalues = [[] for _ in range(len(keys))]
        for i, run in enumerate(self._convergencestats):
            for j in range(len(keys)):
                cvalues[j] += [gen[keys[j]] for gen in run]
        p = plotLineData(cvalues, labelx=labels[0], labely=labels[1], title=title, legend=legend, dot=True)
        self.addPlot(p)

    def plotTrend(self, keyvalues, axislabels, legend, title):
        cvalues = [[],[]]
        for i, run in enumerate(self._convergencestats):
            cvalues[0] += [gen[keyvalues[0]] for gen in run]
            cvalues[1] += [gen[keyvalues[1]] for gen in run]
        polys = []
        datalength = len(cvalues[0])
        for series in cvalues:
            x = [i for i in range(len(series))]
            z = numpy.polyfit(x, series, 3)
            polys.append(numpy.poly1d(z))
        trends = []
        for p in polys:
            trends.append([p(x) for x in range(datalength)])
        p = plotLineData(trends, labelx=axislabels[0], labely=axislabels[1], title=title, legend=legend, xcategorical=True)
        p.legend.border_line_alpha=0
        self.addPlot(p)

    def plotOperatorSuccessTrend(self):
        self.plotTrend(keyvalues=["mutations", "crossovers"], axislabels=["Generation","Success rate trend"], legend = ["Mutations","Crossovers"], title="Operator success rate.")

    def plotOperatorImpactTrend(self):
        self.plotTrend(keyvalues=["mutate_gain", "crossover_gain"], axislabels=["Generation","Mean gain by operator"], legend = ["Mutations","Crossovers"], title="Operator gain in fitness.")

    def plotAlgorithmCost(self):
        cvalues = [[]]
        for i, run in enumerate(self._convergencestats):
            cvalues[0] += [numpy.mean(gen['mean_evaluations']) for gen in run]
        p = plotLineData(cvalues, labelx="Generation", labely="Mean cost of generation", title="Computational cost of algorithm", legend=["Mean Cost"], dot=True)
        self.addPlot(p)

    def plotPareto(self):
        """
        Plot fitness values against complexity in a front
        """
        run = self._convergencestats[-1]
        fitnessvalues = run[-1]['fitness']
        complexity = run[-1]['complexity']
        p = plotFront(X=fitnessvalues, Y=complexity, labelx="Fitness", labely="complexity", title="Front")
        self.addPlot(p)

    def plotAll(self):
        self.plotFitness()
        self.plotComplexity()
        self.plotOperators()
        self.plotOperatorSuccessTrend()
        self.plotOperatorsImpact()
        self.plotOperatorImpactTrend()
        self.plotDepth()
        self.plotAlgorithmCost()
        self.plotFoldingGains()

    def saveData(self, filename, outputfolder=None):
        outputfolder = outputfolder or ""
        with open(outputfolder+filename, 'w') as filename:
            json.dump(self._convergencestats, filename, indent=4)


class SummarizedResults(Plotter):
    def __init__(self, results):
        super().__init__()
        self._results = results
        self.plotAll()

    def plotFitness(self):
        fitness = rmtocm([r['fitness'] for r in self._results])
        p = plotDotData(fitness, labelx="Process", labely="Fitness", title="Fitness of the last generation calculated on the test entire set (sample + test).", xcategorical=True, groupsimilar=True)
        self.addPlot(p)

    def plotDepth(self):
        depth = rmtocm([r['depth'] for r in self._results])
        p = plotDotData(depth, labelx="Process", labely="Depth", title="Depth of the last generation", xcategorical=True, groupsimilar=True, ycategorical=True)
        self.addPlot(p)

    def plotPrediction(self):
        fitness = [r['corr_fitness'] for r in self._results]
        processcount = len(fitness)
        legend = ["Process {}".format(i) for i in range(processcount)]
        p = plotLineData(fitness, labelx="Phase", labely="Correlation (less is better)", title="Correlation between end of phase fitness values on training data and fitness value on full data.", legend=legend, xcategorical=True)
        p.legend.border_line_alpha=0
        self.addPlot(p)

    def plotPredictionTrend(self):
        fitness = [r['corr_fitness'] for r in self._results]
        processcount = len(fitness)
        legend = ["Process {}".format(i) for i in range(processcount)]

        polys = []
        datalength = len(fitness[0])
        for series in fitness:
            x = [i for i in range(len(series))]
            z = numpy.polyfit(x, series, 2)
            polys.append(numpy.poly1d(z))
        trends = []
        for p in polys:
            trends.append([p(x) for x in range(datalength)])
        p = plotLineData(trends, labelx="Phase", labely="Correlation (less is better)", title="Correlation trend between end of phase fitness values on training data and fitness value on full data.", legend=legend, xcategorical=True)
        p.legend.border_line_alpha=0
        self.addPlot(p)

    def plotAll(self):
        self.plotFitness()
        self.plotTrainedFitness()
        self.plotDifference()
        self.plotPrediction()
        self.plotPredictionTrend()
        self.plotDepth()
        self.plotFeatures()

    def plotDifference(self):
        fitness = rmtocm([r['diff_fitness'] for r in self._results])
        p = plotDotData(fitness, labelx="Process", labely="Difference", title="Difference between fitness on sample data and test data per phase best value", groupsimilar=True, xcategorical=True)
        self.addPlot(p)

    def plotTrainedFitness(self):
        fitness = rmtocm([r['last_fitness'] for r in self._results])
        p = plotDotData(fitness, labelx="Process", labely="Fitness", title="Fitness of the last generation calculated on the training set (sample + test).", xcategorical=True, groupsimilar=True)
        self.addPlot(p)

    def plotFeatures(self):
        featurespre = [list(set(r['features'][0])) for r in self._results]
        featurespre = list(map(lambda x : [y+1 for y in x], featurespre))
        mlen = max( [len(x) for x in featurespre])
        #logger.info("Features are {}".format(featurespre))
        for i in range(len(featurespre)):
            x = featurespre[i]
            first = x[0]
            lx = len(x)
            x += [first for _ in range(mlen-lx)]
        features = rmtocm(featurespre)

        #logger.info("Features are {}".format(features))
        p = plotDotData(features, labelx="Process", labely="Features", title="Features used in the best solution.", xcategorical=True, groupsimilar=True, ycategorical=True)
        self.addPlot(p)


    def plotComplexity(self):
        logger.warning("Unused function")
        pass

    def saveData(self, filename, outputfolder=None):
        outputfolder = outputfolder or ""
        with open(outputfolder+filename, 'w') as filename:
            json.dump(self._results, filename, indent=4)
