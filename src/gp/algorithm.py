
from expression.tree import Tree
from gp.population import Population, SetPopulation
from expression.node import Variable
import random


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, seed = None):
        self._population = SetPopulation(key=lambda _tree : 0-_tree.getFitness())
        self._maxdepth = maxdepth
        self._popsize=popsize
        self._seed = seed
        self._rng = random.Random()
        if seed:
            self._rng.seed(seed)
        self._X = X
        self._Y = Y
        self._initialize()

    def _initialize(self):
        vlist = []
        for i, x in enumerate(self._X):
            assert(isinstance(x, list))
            vlist.append(Variable(x, i))
        self._variables = vlist
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
        for i in range(self._popsize):
            t = Tree.growTree(self._variables, self._maxdepth, self._seed+i if self._seed else None)
            t.setFitnessFunction(lambda o : o.getDepth())
            t.updateFitness()
            self.addTree(t)

    def getVariables(self):
        return self._variables

    def printForest(self, prefix):
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")
