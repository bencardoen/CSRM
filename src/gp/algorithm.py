
from expression.tree import Tree
from expression.operators import Mutate, Crossover
from expression.tools import traceFunction
from expression.functions import Constants
from gp.population import Population, SetPopulation
from expression.node import Variable
from copy import deepcopy
from operator import neg
import random
import logging
logger = logging.getLogger('global')


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None, archivesize= None):
        """
        Initializes a forest of trees randomly constructed.

        :param list X: a list of feature values, per feature an equal length sublist
        :param list Y: a list of response values
        :param int popsize: maximum population size
        :param int maxdepth: the maximum depth a tree is initialized to
        :param int generations: generations to iterate
        :param int seed: seed value for the rng used in tree construction
        """
        # Sorted set of trees by fitness value
        self._population = SetPopulation(key=lambda _tree : _tree.getFitness())
        self._datapointcount = len(X[0])
        self._fitnessfunction = fitnessfunction
        self._maxdepth = maxdepth
        self._popsize=popsize
        self._seed = seed
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)
        logging.info(" Data points for X {}".format(X))
        self._X = X
        self._Y = Y
        self._initialize()
        self._archive = SetPopulation(key=lambda _tree : _tree.getFitness)
        self._generations = generations
        self._archivesize = archivesize or self._popsize

    def getSeed(self):
        s = self._seed
        if s is None:
            return None
        self._seed += 1
        return s

    def _initialize(self):
        vlist = []
        assert(len(self._X))
        for i, x in enumerate(self._X):
            assert(isinstance(x, list))
            vlist.append(Variable(x, i))
        self._variables = vlist[:]
        self._initializePopulation()
    

    def addTree(self, t):
        assert(t not in self._population)
        self._population.add(t)

    def getBestTree(self):
        """
        Get Tree with best (highest) fitness score
        """
        return self._population.top()

    def getBestN(self, n, remove=False):
        """
        Get the n fittest trees. Without remove, a view.
        """
        return self._population.getN(n) if not remove else self._population.removeN(n)

    def _initializePopulation(self):
        assert(len(self._variables))
        for i in range(self._popsize):
            logging.error("Growing {}".format(i))
            t = Tree.growTree(self._variables, self._maxdepth, self.getSeed())
            assert(len(t.getVariables()))
            self.addTree(t)
        for t in self._population:
            assert(len(t.getVariables()))
            

    def getVariables(self):
        return self._variables

    def printForest(self, prefix):
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")

    def run(self):
        """
        Main algorithm loop. Evolve population through generations.
        """
        for _ in range(self._generations):
            selected = self.select()
            modified = self.evolve(selected)
            self.update(modified)
            self.archive(modified)
            if self.stopCondition():
                break

    def stopCondition(self):
        """
        Halt algorithm if internal state satisfies some condition
        """
        return False

    @traceFunction
    def select(self):
        """
        Select a subset of the current population to operate on.
        """
        return self._population.removeAll()

    @traceFunction
    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.
        :param list selection: a subset of the population selected
        :return list modified: a subset of modified specimens
        """
        return selection


    @traceFunction
    def update(self, modified):
        """
        Process the new generation.
        """
        return

    @traceFunction
    def archive(self, modified):
        """
        Using the new and previous generation, determine the best specimens and store them.
        """
        return

    def addToArchive(self, t):
        self._archive.add(t)
        if len(self._archive) > self._archivesize:
            self._archive.popLast()

class BruteElitist(GPAlgorithm):
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None):
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed = seed)

    @traceFunction(logcall=logger.info)
    def select(self):
        s = self._population.removeAll()
        assert(len(self._population)==0)
        for t in s:
            assert(len(t.getVariables()))
        return s

    @traceFunction(logcall=logger.info)
    def evolve(self, selection):
        if len(selection) < 2:
            return selection
        for i,t in enumerate(selection):
            logging.info("Evolving {}".format(t.toExpression()))
            Mutate.mutate(t, seed=self.getSeed())
            logging.info("Mutation results in {}".format(t.toExpression()))
            left = t
            right = selection[self._rng.randint(0, len(selection)-1)]
            while right == left:
                right = selection[self._rng.randint(0, len(selection)-1)]
            assert(left != right)
            logging.info("Right selected for crossover {}".format(right.toExpression()))
            Crossover.subtreecrossover(left, right, seed=self.getSeed()) # TODO depth
        return selection

    @traceFunction(logcall=logger.info)
    def update(self, modified):
        # update fitness value here or in evolve ?
        for t in modified:
            oldfit = t.getFitness()
            t.scoreTree(self._Y, self._fitnessfunction)
            newfit = t.getFitness()
            logger.info("Updating \n{}n with fitness {} ---> {}".format(t.toExpression().replace(" ",""), oldfit, newfit))
            self.addTree(t)

    @traceFunction(logcall=logger.info)
    def archive(self, modified):
        t = self.getBestTree()
        t = deepcopy(t)
        self._archive.add(t)
