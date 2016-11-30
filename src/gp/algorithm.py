
from expression.tree import Tree
from gp.population import Population, SetPopulation
from expression.node import Variable
from operator import neg
import random


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None):
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
        self._population = SetPopulation(key=lambda _tree : fitnessfunction(_tree))
        self._maxdepth = maxdepth
        self._popsize=popsize
        self._seed = seed
        self._rng = random.Random()
        if seed:
            self._rng.seed(seed)
        self._X = X
        self._Y = Y
        self._initialize()
        self._archive = SetPopulation(key=lambda _tree : fitnessfunction(_tree))
        self._generations = generations

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

    def select(self):
        """
        Select a subset of the current population to operate on.
        """
        return self._population

    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.
        """
        return selection


    def update(self, modified):
        """
        Process the new generation.
        """
        return

    def archive(self, modified):
        """
        Using the new and previous generation, determine the best specimens and store them.
        """
        return

class BruteElitist(GPAlgorithm):
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None):
        super().__init__(X, Y, popsize, maxdepth, generations, seed)



def _fitfunc(_tree):
    return operators.neg(_tree.getFitness())
