#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree
from expression.operators import Mutate, Crossover
from expression.tools import randomizedConsume, copyObject, consume, pearson as correlator, getRandom
from expression.constants import Constants
from gp.population import SetPopulation
from expression.node import Variable
from meta.optimizer import PSO, PassThroughOptimizer
import random
import logging
import numpy
logger = logging.getLogger('global')
numpy.seterr('raise')


class GPAlgorithm():
    """
    A base class representing a Genetic Programming Algorithm instance.

    The base class is responsible for data structures, configuration and control flow.

    In itself it will not evolve a solution.
    """

    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed=None, archivesize=None, history=None, phases=None, tournamentsize=None, initialdepth=None, skipconstantexpressions=False, archivefile=None):
        """
        Initializes a forest of trees randomly constructed.

        :param list X: a list of feature values, per feature an equal length sublist
        :param list Y: a list of response values
        :param int popsize: maximum population size
        :param int maxdepth: the maximum depth a tree is initialized to
        :param int generations: generations to iterate
        :param int seed: seed value for the rng used in tree construction
        :param int archivesize: size of the archive used between phases to store best-of-generation samples, which are in turn reused in next phases
        :param int tournamentsize: size of subset taken from population (fittest first) to evolve.
        """
        self.pid = None
        self._population = SetPopulation(key=lambda _tree: _tree.getFitness())
        """ Number of entries per feature. """
        self._datapointcount = len(X[0])
        """ Fitness function, passed to tree instance to score."""
        self._fitnessfunction = fitnessfunction
        self._initialdepth = initialdepth or maxdepth
        #logger.info("initialdepth = {}".format(initialdepth))
        self._maxdepth = maxdepth
        self._popsize = popsize
        #logger.info("Using population {} maxdepth {} initdepth {}".format(self._popsize, self._maxdepth, self._initialdepth))
        self._seed = seed
        self._rng = getRandom()
        if seed is not None:
            self._rng.seed(seed)
        else:
            logger.warning("Non deterministic mode.")
        self._trace = False
        """ Input data """
        self._X = X
        #logger.debug("X is {} x {}".format(len(X), len(X[0])))
        #logger.info("Sum X {}".format(sum([sum(x) for x in X])))
        #logger.info("\n\nX is {}\n\n".format(X))
        """ Expected data """
        self._Y = Y
        # logger.info("Y is {} {}".format(len(Y), Y))
        # logger.info("Sum Y is {}".format(sum(Y)))
        self._skipconstantexpressions = skipconstantexpressions
        self._archiveinputfile = archivefile
        if archivefile:
            self._archiveoutputfile = archivefile + "_out" if archivefile else None
        self.archiveoutfile = None
        self._initialize()
        self._archive = SetPopulation(key=lambda _tree: _tree.getFitness())
        self._generations = generations
        self._currentgeneration = 0
        self._archivesize = archivesize or generations
        assert(self._archivesize > 0)
        # List of generation : tuple of stats
        self._convergencestats = []
        """ History is nr of generations the stop condition investigates to determine a valid condition. E.g. : if the last <x> generations have no mutations, stop."""
        self._history = history or 5
        """ Current phase of the algorithm : [0, self._phases). """
        self._phase = 0
        """ Total phases requested """
        self._phases = phases or 1
        """ Size of tournament, determines which samples compete. """
        self._tournamentsize = tournamentsize or popsize
        """ Number of samples to archive between phases. """
        self._archivephase = 4
        """
        Randomizing the selection upon which crossover works can improve the quality of the converged results.
        Non random crossover (e.g. best mates with second best) will lead to faster convergence, albeit to a less optimal solution.
        """
        self._randomconsume = True
        """
        If random consume is active, determine when this should be used (in order to get the best of both)
        """
        self._randomconsumeprobability = 0.5

    @property
    def tournamentsize(self):
        return self._tournamentsize

    @tournamentsize.setter
    def tournamentsize(self, value: int):
        assert(value>0 and value <= self._popsize)
        self._tournamentsize = value

    @property
    def population(self):
        return [ x for x in self._population ]

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value:int):
        assert(value >= 0)
        self._phase = value

    @property
    def phases(self):
        return self._phases

    @property
    def skipconstantexpressions(self):
        return self._skipconstantexpressions

    def getSeed(self):
        """
        Retrieve seed, and modify it for the next call.

        Seed is in [0, 0xffffffff]
        """
        s = self._seed
        if s is None:
            return None
        self._seed = self._rng.randint(0, 0xffffffff)
        return s

    def getArchived(self, n:int):
        # have at least self._archivephase samples
        assert(len(self._archive))
        arch = []
        if n > len(self._archive):
            logger.warning("Requesting more samples than archive contains")
            arch = self._archive.getAll()
        else:
            arch = self._archive.getN(n)
        return [copyObject(a) for a in arch]

    def archiveExternal(self, lst):
        """
        Add x in lst to archive, dropping the worst samples to make place if required.

        Each sample will have its variable set updated to match the rest of the population.
        """
        for x in lst:
            expr = x.toExpression()
            x2 = Tree.createTreeFromExpression(expr, variables=self._X)
            self.addToArchive(x2)

    def addConvergenceStat(self, generation, stat, phase):
        if len(self._convergencestats) <= phase:
            self._convergencestats.append([])
        if len(self._convergencestats[phase]) <= generation:
            self._convergencestats[phase].append(stat)
        else:
            self._convergencestats[generation] = stat

    def getConvergenceStat(self, generation, phase):
        return self._convergencestats[phase][generation]

    def getConvergenceStatistics(self):
        """
        Return all statistics

        Returns a list indiced by phase
        Each sublist is indiced by generation and holds a dict:"fitness", "mean_fitness, "var_fitness", "std_fitness", "replacements" and equivalent for complexity
        """
        return self._convergencestats[:]

    def resetConvergenceStats(self):
        self._convergencestats = []

    def _initialize(self):
        self._variables = [Variable(x, i) for i,x in enumerate(self._X)]
        self._initializePopulation()

    def addTree(self, t):
        #logger.info("Adding tree with id {:0x} and repr {} to pop \{}".format(id(t), t.toExpression(), self._population))
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
        """
        Seed the population with random samples.
        """
        assert(len(self._variables))
        trees = []
        if self._archiveinputfile:
            trees = self.readPopulation(self._archiveinputfile)
            for i in range(min(len(trees), self._popsize)):
                t = trees[i]
                self._population.add(t)
        diffsize = self._popsize - len(self._population)
        assert(diffsize >= 0)
        for i in range(diffsize):
            self.addRandomTree()

    def addRandomTree(self):
        """
        Create a random tree using this algorithm's configuration.

        The generated tree is guaranteed to be viable (i.e. has a non inf fitness)
        The generation process will skip constant expressions.
        """
        rng = self._rng
        t = Tree.growTree(self._variables, self._initialdepth, rng=rng)
        t.scoreTree(self._Y, self._fitnessfunction)
        # Minfitness --> indicates that for at least one data point the instance is not semantically valid
        # We can use the algorithm to filter them out, but this only loads the operators even more.
        # Next test if we allready have an equivalent function in the population. For each set of points a
        # near infinite set of approximating functions exist, there is no gain in addding a function that has the exact
        # same approximation (barring filtering on features).
        # Finally, filter out constant expressions. In a non trivial discovery we don't need ctexprs, they
        # cause an increased load for the algorithm. It's still possible ctexprs are generated by mutation or
        # crossover, but we can use constant folding there.
        ctexprcheck = self.skipconstantexpressions
        while t.getFitness() == Constants.MINFITNESS or (t in self._population) or (ctexprcheck and t.isConstantExpressionLazy()):
            assert(self._variables)
            t = Tree.growTree(self._variables, self._initialdepth, rng=rng)
            t.scoreTree(self._Y, self._fitnessfunction)

        if ctexprcheck:
            assert(not t.isConstantExpressionLazy())
        self.addTree(t)

    def testInvariant(self):
        assert(len(self._population) == self._popsize)
        if self._maxdepth is not None:
            for d in self._population:
                assert(d.getDepth() <= self._maxdepth)

    def getVariables(self):
        return self._variables

    def printForestToDot(self, prefix):
        """
        Write out the entire population to .dot files with prefix
        """
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")

    def summarizeSamplingResults(self, X, Y):
        assert(len(X[0]) == len(Y))
        for t in self._population:
            t.updateVariables(X)
            t.scoreTree(Y, self._fitnessfunction)
        try:
            fit = [d.getFitness() if d.getFitness()!= Constants.MINFITNESS else Constants.PEARSONMINFITNESS for d in self._population]
            scoredpopulation = [(f, t.toExpression()) for f, t in zip(fit, self._population)]
            features = [d.getFeatures() for d in self._population]
            depths = [d.getDepth() for d in self._population]
            comp = [d.getScaledComplexity() for d in self._population]
            mean, sd, v= numpy.mean(fit), numpy.std(fit), numpy.var(fit)
            cmean, csd, cv = numpy.mean(comp), numpy.std(comp), numpy.var(comp)
            lastfit = [self.getConvergenceStat(-1, phase)['fitness'] for phase in range(self.phases)]
            cfit = [correlator(fit, best) for best in lastfit]
            assert(len(cfit) == self.phases)
            dfit = [abs(a-b) for a,b in zip(lastfit[-1], fit)]
            dmeanfit, dsdfit, dvfit = numpy.mean(dfit), numpy.std(dfit), numpy.var(dfit)
            logger.info("Best fitness value for full data is {} mean {} sd {} var {}".format(min(fit), mean, sd, v))

        except FloatingPointError as e:
            logger.error("Floating point error on values fit {} ".format(fit))
            raise e

        return {"fitness":fit,"mean_fitness":mean, "std_fitness":sd, "variance_fitness":v, "depth":depths,
                "mean_complexity":cmean, "std_complexity":csd, "variance_complexity":cv,"complexity":comp,
                "corr_fitness":cfit, "diff_mean_fitness":dmeanfit, "diff_std_fitness":dsdfit, "diff_variance_fitness":dvfit,
                "diff_fitness":dfit, "features":features, "last_fitness":lastfit[-1], "solution":scoredpopulation}

    def getPopulation(self):
        return [p.toExpression() for p in self._population]

    def summarizeGeneration(self, replacementcount:list, mutategain:list, crossovergain:list, evaluations:list, optimizergains:dict, generation:int, phase:int):
        """
        Compute fitness statistics for the current generation and record them
        """
        fit = [min(d.getFitness(),Constants.PEARSONMINFITNESS) for d in self._population ]
        depths = [d.getDepth() for d in self._population]
        comp = [d.getScaledComplexity() for d in self._population]
        mean, sd, v= 0,0,0
        cmean, csd, cv = 0,0,0
        try:
            mean, sd, v= numpy.mean(fit), numpy.std(fit), numpy.var(fit)
            cmean, csd, cv = numpy.mean(comp),numpy.std(comp), numpy.var(comp)
        except FloatingPointError as e:
            logger.error("FPE {} for {}".format(e,fit))
        mg = list(filter(lambda x : x > 0, mutategain))
        mmg = 0
        if mg:
            mmg = numpy.mean(mg)
        cg = list(filter(lambda x : x > 0, crossovergain))
        mcg = 0
        if cg:
            mcg = numpy.mean(cg)
        meaneval = numpy.mean(evaluations)
        fsavings = optimizergains["foldingsavings"]
        nc = optimizergains["nodecount"]
        constantfoldingsavings = fsavings/nc * 100
        fitnessgains = optimizergains["fitnessgains"]

        assert(isinstance(replacementcount, list))
        #logger.debug("Generation {} SUMMARY:: fitness \tmean {} \tsd {} \tvar {} \treplacements {}".format(generation, mean, sd, v, replacementcount[0]))

        self.addConvergenceStat(generation, {    "fitness":fit,"mean_fitness":mean, "std_fitness":sd, "variance_fitness":v, "depth":depths,
                                                 "replacements":replacementcount[0],"mutations":replacementcount[1], "crossovers":replacementcount[2],
                                                 "mean_complexity":cmean, "std_complexity":csd, "variance_complexity":cv,"complexity":comp,
                                                 "mutate_gain":mmg, "crossover_gain":mcg, "mean_evaluations":meaneval, "foldingsavings":constantfoldingsavings, "fitnessgains":fitnessgains}, phase)

    def setTrace(self, v, prefix):
        """
        Enables generation per generation tracing (e.g. writing to dot)
        """
        self._trace = v
        self._prefix = prefix

    def printForest(self):
        """
        Write out population in ASCII
        """
        print(str(self._population))

    def reseed(self):
        """
        After a run of x generations, reseed the population based on the archive
        """
        #logger.info("Reseeding")
        archived = self._archive.getAll()
        for a in archived:
            #a.scoreTree(self._Y, self._fit)
            self.addTree(copyObject(a))
        # Retrim the current population by removing the least fit samples
        #logger.info("Population after archive usage is {}".format([t.getFitness() for t in self._population]))
        while len(self._population) > self._popsize:
            self._population.drop()
        # If we have too few, refill with random samples
        diff = self._popsize - len(self._population)
        for _ in range(diff):
            self.addRandomTree()
        #logger.info("After reseeding population is {}".format([t.getFitness() for t in self._population]))

    def writePopulation(self, filename):
        with open(filename, 'w') as f:
            for p in self._population:
                f.write(p.toExpression() + "\n")

    def readPopulation(self, filename):
        trees = []
        try:
            with open(filename, 'r') as f:
                trees = []
                lines = f.readlines()
                for l in lines:
                    #logger.info("Reading {}".format(l))
                    t = None
                    try:
                        t = Tree.createTreeFromExpression(l[:-1], variables=self._X)
                        t.scoreTree(self._Y, self._fitnessfunction)
                        if t.getFitness() == Constants.MINFITNESS:
                            logger.warning("Stored tree sample with expression {} gave invalid fitness value on new data, skipping.".format(l))
                        else:
                            #logger.info("Accepted stored tree sample.")
                            trees.append(t)
                    except ValueError as e:
                        logger.error("Failed parsing instance expression {}".format(l))
        except FileNotFoundError as e:
            logger.error("Archive file not found {}".format(filename))
        return trees


    def restart(self):
        """
        Called before a new phase runs, clear the last population and reseeds the new using the archive.
        """
        self._currentgeneration = 0
        self._population.removeAll()
        self.reseed()

    def run(self):
        """
        Main algorithm loop. Evolve population through generations.

        This loop is the main control flow of the algorithm, subclasses can simply override
        called methods to alter behavior

        """
        r = self._phase
        for i in range(self._generations):
            selected = self.select()
            modified, count, gm, gc, evaluations = self.evolve(selected)
            #logger.info("Modified is {} of type {} and length {}".format(modified, type(modified), len(modified)))
            gains = self.optimize(modified)
            self.update(modified)
            assert(isinstance(count, list))
            self._currentgeneration = i
            self.summarizeGeneration(count, gm, gc, evaluations, gains,generation=i, phase=r)
            if self.stopCondition():
                logger.info("Stop condition triggered")
                break
            self.testInvariant()
            if self._trace:
                self.printForestToDot("process_{}_phase_{}_generation_{}".format(self.pid, self._phase, i))
        self._phase += 1
        self.archive(modified)

    def executeAlgorithm(self):
        self._phase = 0
        self.run()
        for i in range(self._phases-1):
            self.restart()
            logger.info("----Phase {}".format(i+1))
            self.run()

    def stopCondition(self):
        """
        Halt algorithm if internal state satisfies some condition
        """
        return False

    def select(self):
        """
        Select a subset of the current population to operate on.
        """
        assert(len(self._population))
        #logger.debug("Selecting {} from {}".format(self._tournamentsize, self._population))
        sel = self._population.removeN(self._tournamentsize)
        #logger.debug("Selected {} ".format(sel))
        assert(len(sel) == self._tournamentsize)
        assert(len(self._population) == self._popsize - self._tournamentsize)
        return sel

    def evaluate(self, sel=None):
        """
        Recalculate the fitness of the population if sel == None, else the selection.
        """
        if sel is None:
            sel = self._population
        for t in sel:
            t.scoreTree(self._Y, self._fitnessfunction)

    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.

        :param list selection: a subset of the population selected
        :return tuple modified: a tuple of modified samples based on selection, and changes made
        """
        self.evaluate(selection)
        return selection, [0,0,0], [], [], []

    def requireMutation(self, popindex:int)->bool:
        """
        Decide if mutation is desirable
        """
        return True

    def minDepthRatio(self, popindex:int):
        return 0

    def optimize(self, selected):
        """
        Apply a metaheuristic to the selection (in place).
        """
        totalnodes = sum([t.nodecount for t in selected])
        gain = {"nodecount":totalnodes}
        gain["foldingsavings"] = 0
        gain["fitnessgains"] = [0 for t in selected]
        return gain

    def update(self, modified):
        """
        Process the new generation.

        At the very least, adds modified back to population based on a condition.
        """
        assert(len(modified) == self._tournamentsize)
        assert(len(self._population) == self._popsize-self._tournamentsize)
        for t in modified:
            if t not in self._population:
                self.addTree(t)
        remcount = self._popsize - len(self._population)
        if remcount > 0:
            for _ in range(remcount):
                self.addRandomTree()
        else:
            for _ in range(- remcount):
                self._population.drop()

    def archive(self, modified):
        """
        Using the new and previous generation, determine the best specimens and store them.
        """
        logger.warning("Inactive archive function called")
        raise NotImplementedError

    def addToArchive(self, t):
        """
        Add t to the archive and truncate worst if necessary.
        """
        #logger.info("Adding {} to archive".format(t.getFitness()))
        if t.getFitness() == Constants.MINFITNESS:
            #logger.info("Invalid communicated sample, ignoring.")
            return
        if t not in self._archive:
            self._archive.add(t)
            if len(self._archive) > self._archivesize:
                #logger.info("Archive overflowing.")
                self._archive.drop()
        else:
            pass
            # logger.warning("Sample {} already in archive!".format(t.getFitness()))
            # logger.warning("Archive is {}".format([t.getFitness() for t in self._archive]))


class BruteElitist(GPAlgorithm):
    """
    Brute force Elitist GP Variant.

    Applies mutation and subtree crossover to entire population and aggresively
    replaces unfit samples.
    """

    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations, seed=None, phases=None, archivesize=None, initialdepth=None, skipconstantexpressions=False, archivefile=None):
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed=seed, phases=phases, archivesize=archivesize, initialdepth=initialdepth, skipconstantexpressions=skipconstantexpressions, archivefile=archivefile)

    def evolve(self, selection):
        """
        Evolve the current generation into the next.

        Apply mutation on each sample, replacing if fitter.
        Apply subtree crossover using random selection of pairs, replacing if fitter.
        """
        Y = self._Y
        fit = self._fitnessfunction
        d = self._maxdepth
        rng = self._rng
        variables = self._variables

        # successful total, mutations, crossover
        replacementcount = [0,0,0]
        # total
        operationcount = [0,0,0]
        selcount = len(selection)
        assert(selcount == self._tournamentsize)

        # Mutate on entire population, with regard to (scaled) fitness
        # Replacement is done based on comparison t, t'. It's possible in highly varied populations that this strategy
        # is not ideal.
        gainsmutate = []
        evaluations = []
        for i in range(0, selcount):
            t = selection[i]
            if self.requireMutation(i):
                candidate = copyObject(t)
                mdr = self.minDepthRatio(i)
                Mutate.mutate(candidate, variables=variables, equaldepth=False, rng=rng, limitdepth=d, mindepthratio=mdr)
                candidate.scoreTree(Y, fit)
                operationcount[0] += 1
                operationcount[1] += 1
                evaluations.append(candidate.evaluationcost)
                of = min( t.getFitness(), Constants.PEARSONMINFITNESS) # in parallel, it's possible a sample has infinite fitness
                nf = min(candidate.getFitness(), Constants.PEARSONMINFITNESS)
                gain = of - nf
                gainsmutate.append(gain)
                if gain > 0:
                    assert(candidate.getDepth() <= d)
                    selection[i] = candidate
                    replacementcount[0] += 1
                    replacementcount[1] += 1

        # Subtree Crossover
        # Select 2 random trees, crossover, if better than parent, replace
        # Crossover disseminates potentially 'good' subtrees, at the cost of diversity
        # Experiments with only applying it to the best specimens increase mean fitness
        # Both fit and unfit individuals benefit from crossbreeding.
        gainscrossover = []
        newgen = []
        if selcount % 2:
            newgen.append(selection[0])
            del selection[0]

        selector = None
        if self._randomconsume:
            toss = self._rng.random()
            if toss >= self._randomconsumeprobability:
                selector = randomizedConsume(selection, seed=self.getSeed())
        selector = selector if selector is not None else consume(selection)

        while selection:
            left = next(selector)
            right = next(selector)
            if id(left) == id(right):
                logger.warning("Left = {}\n Right = {}\n, Selection = {}".format(left, right, selection))
            assert(id(left) != id(right))
            lc = copyObject(left)
            rc = copyObject(right)
            mdr = self.minDepthRatio(i)
            Crossover.subtreecrossover(lc, rc, depth=None, rng=rng, limitdepth=d, mindepthratio=mdr)
            operationcount[0] += 2
            operationcount[2] += 2
            lc.scoreTree(Y, fit)
            evaluations.append(lc.evaluationcost)
            evaluations.append(rc.evaluationcost)
            rc.scoreTree(Y, fit)
            scores = [left, right, lc, rc]
            best = sorted(scores, key = lambda t:t.getFitness())[0:2]
            if lc in best:
                replacementcount[0] += 1
                replacementcount[2] += 1
            if rc in best:
                replacementcount[0] += 1
                replacementcount[2] += 1

            oldfit = min(left.getFitness(), Constants.PEARSONMINFITNESS) + min(right.getFitness(), Constants.PEARSONMINFITNESS)
            nextfit = sum([x.getFitness() if x.getFitness() != Constants.MINFITNESS else Constants.PEARSONMINFITNESS for x in best])
            gain = oldfit - nextfit
            newgen += best
            gainscrossover.append(gain)
        assert(len(newgen) == selcount)
        replacementratio = [x/y if y!=0 else 0 for x, y in zip(replacementcount, operationcount)]
        return newgen, replacementratio, gainsmutate, gainscrossover, evaluations

    def stopCondition(self):
        """
        Composite stop condition.

        Looks at x past generations (set by self._history), then decides if convergence has stalled (std deviation < Constants.FITNESS_EPSILON).
        If std deviation is still large enough, do a check of successful replacements. If no operator in the last x generation was able to generate
        a single fitter sample, then obviously fitness will not improve any further.
        :returns: True if algorithm should halt after this generation.
        """
        generations = len(self._convergencestats[self._phase])
        if generations < self._history:
            return False

        # We want to return True if std_fitness < Epsilon for the last x generations
        found = False
        for i in range(self._history):
            if self.getConvergenceStat(generations-i-1, self._phase)['std_fitness'] > Constants.FITNESS_EPSILON:
                found=True
                break
        if found:  # Still enough variation
            for i in range(self._history):
                if self.getConvergenceStat(generations-i-1, self._phase)['replacements'] != 0:
                    return False  # enough variation, and replacements
            # Replacements are no longer taking place, with enough variation, stop
        return True  # No more variation, stop

    def archive(self, modified):
        """
        Simple archiving strategy, get best of generation and store.
        """
        best = self._population.getN(self._archivephase)
        for b in best:
            self.addToArchive(copyObject(b))


class BruteCoolingElitist(BruteElitist):
    """
    Uses a cooling strategy to apply operators, maximizing gain in the initial process but reducing cost when gain is no longer possible.

    The cooling schedule 'predicts' efficiency of the operators.

    The optimizer if, passed, will be applied to the constants in the generated expressions.

    :param optimizer: class name of optimizer to use
    :param optimizestrategy: 0 (None), 1 (Best only), k (k best), populationcount (all).
    """

    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations, seed=None, phases=None, archivesize=None, initialdepth=None,depthcooling=False, skipconstantexpressions=False, archivefile=None,optimizer=None, optimizestrategy=None):
        self._depthcooling = depthcooling
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed=seed, phases=phases, archivesize=archivesize, initialdepth=initialdepth, skipconstantexpressions=skipconstantexpressions, archivefile=archivefile)
        self.optimizer = optimizer
        self.optimizestrategy = optimizestrategy if optimizestrategy is not None else 1
        logger.info("optimizestrategy {}".format(self.optimizestrategy))

    @property
    def depthcooling(self):
        return self._depthcooling

    def requireMutation(self, popindex:int)->bool:
        generation = self._currentgeneration
        generations = self._generations
        ranking = popindex
        population = self._popsize
        rng = self._rng
        return probabilityMutate(generation, generations, ranking, population, rng=rng)

    def minDepthRatio(self, popindex):
        if self.depthcooling:
            return coolingMinDepthRatio(self._currentgeneration, self._generations, popindex, self._popsize, rng=self._rng)
        else:
            return 0

    def optimize(self, selected):
        """
        Apply a metaheuristic to the selection (in place).

        Will apply constant folding to make the optimizing step more efficient.
        :returns gain: statistics object recording gains.
        """
        totalnodes = sum([t.nodecount for t in selected])
        gain = {"nodecount":totalnodes, "optimizercost":0, "fitnessgains":[0 for t in selected], "fitnessgainsrelative":[0 for t in selected], "foldingsavings":0}
        j = 0
        if self.optimizer and self.optimizestrategy > 0:
            logger.info("Using optimizer")
            for t in selected:
                if t.getValuedConstants():
                    oldf = t.getFitness()
                    gain["foldingsavings"] += t.doConstantFolding()
                    opt = self.optimizer(populationcount = 50, particle=copyObject(t), distancefunction=self._fitnessfunction, expected=self._Y, seed=0, iterations=50)
                    opt.run()
                    sol = opt.getOptimalSolution()
                    gain["optimizercost"] += sol["cost"]
                    best = sol["solution"]
                    tm = copyObject(t)
                    tm.updateValues(best)
                    tm.scoreTree(self._Y, self._fitnessfunction)
                    newf = tm.getFitness()
                    fgain = oldf - newf
                    if fgain < 0:
                        # It's possible the initial perturbation disturbs the optimizer enough to cause this behavior
                        pass
                    else:
                        t.updateValues(best)
                        t.scoreTree(self._Y, self._fitnessfunction)
                        gain["fitnessgains"].append(fgain)
                        gain["fitnessgainsrelative"].append(fgain/oldf)
                    j += 1
                    if j > self.optimizestrategy:
                        logger.info("Cutoff reached, skipping optimize step, {} > {}".format(j, self.optimizestrategy))
                break
        else:
            #logger.warning("No optimizer set, skipping.")
            pass
        return gain

    def optimizeBest(self, selected):
        totalnodes = sum([t.nodecount for t in selected])
        gain = {"nodecount":totalnodes, "optimizercost":0, "fitnessgains":[0 for t in selected], "fitnessgainsrelative":[0 for t in selected], "foldingsavings":0}
        for t in selected:
            oldf = t.getFitness()
            gain["foldingsavings"] += t.doConstantFolding()
            opt = self.optimizer(populationcount = 50, particle=copyObject(t), distancefunction=self._fitnessfunction, expected=self._Y, seed=0, iterations=50)
            opt.run()
            sol = opt.getOptimalSolution()
            gain["optimizercost"] += sol["cost"]
            best = sol["solution"]
            tm = copyObject(t)
            tm.updateValues(best)
            tm.scoreTree(self._Y, self._fitnessfunction)
            newf = tm.getFitness()
            fgain = oldf - newf
            if fgain < 0:
                # It's possible the initial perturbation disturbs the optimizer enough to cause this behavior
                pass
            else:
                t.updateValues(best)
                t.scoreTree(self._Y, self._fitnessfunction)
                gain["fitnessgains"].append(fgain)
                gain["fitnessgainsrelative"].append(fgain/oldf)
        # Store gain

    def archive(self, modified):
        """
        Simple archiving strategy, get best of generation and store.
        """
        #logger.info("Optimizer archiving")
        best = self._population.getN(self._archivephase)
        best = [copyObject(b) for b in best]
        if self.optimizestrategy == 0:
            #logger.info("Archive optimization running.")
            self.optimizeBest(best)
        for b in best:
            self.addToArchive(b)



def probabilityMutate(generation:int, generations:int, ranking:int, population:int, rng:random.Random=None)->bool:
    """
    Generate a probability to mutate based on 2 parameters : the current generation and the the ranking of the sample.

    A fitter sample (either by age (generation) or rank) is less likely to gain from mutation, and will even lose fitness.
    The probability that random information introduced by mutation leads to a better result compared with evolved information decreases.
    This function estimates this probability, thereby avoiding unnecessary mutations and surprisingly leading to faster convergence.

    :param int ranking: index in a fitness (best first) sorted array

    :param int generation: the current generation

    :param int generations: total generations allowed

    :param int population: total nr of specimens

    :param random.Random rng: to reproduce results (optional)

    :returns bool: true if mutation is considered beneficial
    """
    if rng is None:
        logger.warning("Non deterministic mode")
        rng = getRandom()
    q = (generation / generations) * 0.5
    w = (ranking / population) * 2
    r = rng.uniform(a=0,b=1)
    s = rng.uniform(a=0, b=1)
    return r > q and s < w


def coolingMinDepthRatio(generation:int, generations:int, ranking:int, population:int, rng: random.Random=None):
    return generation / generations
