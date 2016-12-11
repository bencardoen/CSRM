#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from analysis.plot import plotDotData, displayPlot, plotFront
from expression.tools import rmtocm
import logging
logger = logging.getLogger('global')

class Convergence:
    def __init__(self, convergencestats):
        self._convergencestats = convergencestats
        self._runs = len(self._convergencestats)

    def plotFitness(self):
        """
            Plot fitness values over the generations
        """
        run = self._convergencestats[0]
        fitnessvalues = [gen['fitness'] for gen in run]
        # Fitness values is a list for each generation
        generations = len(run)
        converted = rmtocm(fitnessvalues)
        logger.info("Got {} generations".format(generations))
        p = plotDotData(converted, labelx="Generation", labely="Fitness", title="Fitness")
        displayPlot(p)

    def plotComplexity(self):
        """
            Plot complexity values over the generations
        """
        run = self._convergencestats[0]
        complexity = [gen['complexity'] for gen in run]
        generations = len(run)
        converted = rmtocm(complexity)
        logger.info("Got {} generations".format(generations))
        p = plotDotData(converted, labelx="Generation", labely="Complexity", title="Complexity")
        displayPlot(p)

    def plotPareto(self):
        """
            Plot fitness values against complexity in a front
        """
        # get last fitness values
        # get last complexity values
        # Plot X as fitness
        # Plot Y as complexity
        run = self._convergencestats[-1]
        fitnessvalues = run[-1]['fitness']
        complexity = run[-1]['complexity']
        logger.info("Plotting fitness {} against complexity {}".format(fitnessvalues, complexity))
        p = plotFront(X=fitnessvalues, Y=complexity, labelx="Fitness", labely="complexity", title="Front")
        displayPlot(p)
