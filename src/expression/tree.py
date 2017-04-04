#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


from expression.tools import msb, compareLists, traceFunction, copyObject, getRandom, flatten
from expression.functions import functionset, getRandomFunction, tokenize, infixToPostfix, isFunction, isOperator, infixToPrefix, Constants
from copy import deepcopy
from .node import Node, Constant, ConstantNode, Variable, VariableNode
import logging
import random
import math
from itertools import islice
from sortedcontainers import SortedDict

# Configure the log subsystem
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')


class Tree:
    """
    Expression tree. A binary tree stored as an ordered list.

    Internal nodes hold operators, terminal nodes either Variables or Constants.
    Each node has an optional constant weight
    """

    def __init__(self):
        # List of nodes in order of generation
        self.nodes = SortedDict()
        self.root = None
        # Cached evaluation
        self.evaluated = 0
        self.modified = False
        self.modifiedDepth = False
        self.depth = None
        self.fitness = Constants.MINFITNESS
        self.fitnessfunction = None
        self._datapointcount = None

    def setDataPointCount(self):
        v = self.getVariables()
        if v is None or len(v) == 0:
            self._datapointcount = 0
            return
        self._datapointcount = len(v[0])

    # Equality of a tree is defined based on its fitness measure.
    # This means that the optimization algorithm will (try) not to keep
    # samples with identical fitness values. This isn't always avoidable
    def __hash__(self):
        return hash(self.fitness)

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __neq__(self, other):
        return self != other

    def getDataPointCount(self):
        if self._datapointcount is None:
            raise ValueError("Invalid datapointcount state !!, Tree must be configured with variables first")
        return self._datapointcount

    def getNode(self, pos: int):
        if pos in self.nodes:
            return self.nodes[pos]
        else:
            return None

    def getFitness(self):
        """
        Get this tree object's absolute fitness value (based on a distance measure)
        """
        return self.fitness

    def getMultiObjectiveFitness(self):
        f = Constants.COMPLEXITY_WEIGHT*self.getScaledComplexity() + Constants.FITNESS_WEIGHT*self.getFitness()
        assert(f <= 1 or f == Constants.MINFITNESS)
        return f

    def setFitness(self, fscore):
        self.fitness = fscore

    def testInvariant(self):
        """
        Ensure the tree is still correct in structure
        """
        self.setDataPointCount()
        V = self.getVariables()
        for v in V:
            assert(len(v.getValues()) == self.getDataPointCount())

    def _addNode(self, node: Node, pos: int):
        """
        Add node to tree (without linking), insert variable if a terminal node.
        """
        self.modified = True
        self.modifiedDepth=True
        if pos == 0:
            self.root = node
        # curlen = len(self.nodes)
        # if pos >= curlen:
        #     self.nodes.extend([None] * (pos-curlen + 1))
        # if self.nodes[pos]:
        #     raise ValueError("Node exists at {}".format(pos))
        # else:
        #     self.nodes[pos] = node
        if pos in self.nodes:
            raise ValueError("Node exists at {}".format(pos))
        self.nodes[pos] = node

    def makeInternalNode(self, function, parent=None, constant=None):
        """
        Add an internal node to the tree
        """
        position = 0
        n = Node(function, position, constant)
        if parent is not None:
            children = parent.getChildren()
            position = children[-1].getPosition()+1 if children else parent.getPosition()*2 + 1
            n.setPosition(position)
            parent.addChild(n)
        else:
            self.root = n
        self._addNode(n, position)
        return n

    def _getDatapointCount(self):
        assert(False)

    def getNodes(self):
        #return [self.root] + self.root.getAllChildren()
        # TODO switch to generator
        #return [n for n in self.nodes if n]
        return list(self.nodes.values())

    def isLeaf(self, position: int):
        """
        Return true if node at position is a terminal/leaf node.
        """
        node = self.getNode(position)
        return node.isLeaf()

    def makeLeaf(self, variable, parent: Node, constant = None):
        """
        Make a leaf node holding a reference to a variable.
        """
        if parent is None:
            n = VariableNode( 0, variable, constant)
            self.root = n
            self._addNode(n, 0)
            return n
        else:
            children = parent.getChildren()
            position = 0
            if children:
                position = children[-1].getPosition()+1
            else:
                position = parent.getPosition()*2+1
            n = VariableNode( position, variable, constant)
            parent.addChild(n)
            self._addNode(n, position)
            return n

    def makeConstant(self, constant, parent: Node):
        """
        Make a leaf node holding a constant
        """
        if parent is None:
            n = ConstantNode( 0, constant)
            self._addNode(n, 0)
            return n
        else:
            children = parent.getChildren()
            position = 0
            if children:
                position = children[-1].getPosition()+1
            else:
                position = parent.getPosition()*2+1
            n = ConstantNode( position, constant)
            parent.addChild(n)
            self._addNode(n, position)
            return n

    def __str__(self):
        root = self.nodes[0]
        output = "Fitness {}\n".format(self.getFitness())
        output += str(root) + "\n"
        children = root.getChildren()[:]
        while len(children) != 0:
            gchildren = []
            for c in children:
                output += str(c) + "\t"
                gchildren += c.getChildren()[:]
            children = gchildren
            output += "\n"
        return output

    def evaluateTree(self, index = None):
        """
        Evaluates tree if tree was modified, else returns a cached results.

        Updates depth if needed.
        """
        if self.modified or index is not None:
            self.evaluated = Node.evaluateAsTree(self.root, index)
            self.modified = False
        return self.evaluated

    def evaluateAll(self):
        """
        For each data point, evaluate this tree object.

        :returns list : list of evaluations
        """
        dpoint = self.getDataPointCount()
        if dpoint != 0:
            return [self.evaluateTree(index=i) for i in range(dpoint)]
        else:
            # TODO use as ctexpr detection
            #logger.info(" Constant Tree is {}".format(self.toExpression()))
            return [self.evaluateTree(index=None)]

    def scoreTree(self, expected, distancefunction):
        """
        Evaluate a tree w.r.t a distancefucntion.

        :param expected list: list A set of expected output values for each datapoint.
        :param distancefunction: Calculates a measure between the calculated and expected values, signature = f(actual, expected, tree)
        :returns float: A fitness value, with lower indicating better. Distancefunction determines the range and scale of the result.
        """
        actual = self.evaluateAll()
        if self.getDataPointCount() == 0:
            #logger.warning("Constant expression, repeating results")
            actual = [actual[0] for _ in range(len(expected))]
        f = distancefunction(actual, expected, tree=self)
        self.setFitness(f)
        return f

    def printToDot(self, name = None):
        filename = name or "output.dot"
        handle = open(filename, 'w')
        handle.write("digraph BST{\n")
        nodes = self.getNodes()
        for c in nodes:
            if c:
                logging.debug("Writing {} to dot".format(c))
                handle.write( str( id(c) ) + "[label = \"" + str(c) + "\"]" "\n")
        for n in nodes:
            if n:
                for c in n.getChildren():
                    handle.write(str(id(n)) + " -> " + str(id(c)) + ";\n")
        handle.write("}\n")
        handle.close()

    @property
    def nodecount(self):
        return len(list(filter(lambda x: x is not None, self.nodes)))

    @property
    def lastposition(self):
        # todo optimize
        return self.getNodes()[-1].getPosition()

    @property
    def evaluationcost(self):
        return self.getComplexity()

    def getDepth(self):
        """
        Return depth of this tree.

        Returns a cached version or calculates a new one if needed.
        """
        if self.modifiedDepth:
            self.depth = self.calculateDepth()
            self.modifiedDepth=False
        return self.depth

    def calculateDepth(self):
        return self.getNodes()[-1].getDepth()
        # for n in reversed(self.nodes):
        #     if n :
        #         return n.getDepth()
        # raise ValueError("No Nodes in tree")

    @staticmethod
    def makeRandomTree(variables, depth: int, rng=None, tokenLeafs=False, limit=None):
        """
        Generate a random expression tree with a random selection of variables.

        Topdown construction, there is no guarantee that this construction renders a semantically valid tree

        :param bool tokenLeafs: if set, only use constants
        :param int limit: raises Exception if no valid tree can be found in at least limit tries
        :returns expression.tree.Tree: A randomized tree.
        """
        _rng = rng
        if rng is None:
            logger.warning("Using non deterministic mode")
            _rng = getRandom()
        if depth == 0:
            t = Tree()
            if (_rng.randrange(0, 2) & 1) and variables:
                child = t.makeLeaf(_rng.choice(variables), None)
            else:
                child = t.makeConstant(Constant.generateConstant(rng=_rng), None)
            return t
        cnt = 0
        while True:
            t= Tree()
            nodes = [t.makeInternalNode(getRandomFunction(rng=_rng), None, None)]
            for i in range(depth):
                newnodes=[]
                addnode = newnodes.append
                for node in nodes:
                    for j in range(node.getArity()):
                        if i >= depth-1:
                            if tokenLeafs:
                                child = t.makeConstant(Constant(1.0), node)
                            else:
                                if (_rng.randrange(0, 2) & 1) and variables:# todo bias features
                                    child = t.makeLeaf(_rng.choice(variables), node)
                                else:
                                    child = t.makeConstant(Constant.generateConstant(rng=_rng), node)
                        else:
                            child = t.makeInternalNode(getRandomFunction(rng=_rng), node, None)
                            addnode(child)
                nodes = newnodes
            e = t.evaluateTree()
            if e is None:
                cnt += 1
                if limit is not None:
                    if cnt >= limit:
                        raise ValueError("Invalid iteration count")
            else:
                t.setDataPointCount()
                return t

    @staticmethod
    def constructFromSubtrees(left, right, rng=None):
        """
        Construct a valid tree from left and right subtrees.
        """
        if rng is None:
            logger.warning("Using non deterministic mode")
            rng = getRandom()
        while True:
            t = Tree.makeRandomTree(variables=None, depth=1, rng=rng, tokenLeafs=True)
            if t.getRoot().getArity() == 2:
                t.spliceSubTree(t.getNode(1), left.getRoot())
                t.spliceSubTree(t.getNode(2), right.getRoot())
            else:
                t.spliceSubTree(t.getNode(1), left.getRoot())
            e = t.evaluateTree()
            if e is not None:
                t.setDataPointCount()
                return t


    @staticmethod
    def growTree(variables, depth: int, rng=None):
        """
        Grow a tree up to depth.
        """
        if rng is None:
            logger.warning("Using non deterministic mode")
            rng = getRandom()
        if depth <= 1:
            return Tree.makeRandomTree(variables=variables, depth=depth, rng=rng)
        return Tree._growTree(variables=variables, depth=depth, rng=rng)

    @staticmethod
    def _growTree(variables, depth: int, rng=None):
        left = None
        right = None
        if depth == 2:
            left = Tree.makeRandomTree(variables, depth=1, rng=rng)
            right = Tree.makeRandomTree(variables, depth=1, rng=rng)
        else:
            left = Tree._growTree(variables, depth-1, rng=rng)
            right = Tree._growTree(variables, depth-1, rng=rng)
        root = Tree.constructFromSubtrees(left, right, rng=rng)
        return root

    def getRandomNode(self, seed = None, depth = None, rng=None, mindepth=None):
        """
        Return a randomly selected node from this tree.

        If parameter depth is set, select only nodes at that depth

        :returns: a selected node, never root
        """
        assert(isinstance(seed, int) or seed is None)
        #logger.info("RNG = {} SEED = {}".format(rng, seed))
        ln = self.lastposition
        #logger.info("LSN = {} ".format(ln))
        if depth is not None:
            assert(depth <= self.getDepth())
        r = rng or getRandom()
        if seed is not None:
            r.seed(seed)
        if seed is None and rng is None:
            logger.warning("Non deterministic mode")
        lowerrange = 1
        if mindepth is not None:
            lowerrange=(2**mindepth) - 1
            assert(depth is None)
        lower, upper = lowerrange, ln+1
        if depth:
            assert(depth < math.log(ln+2, 2))
            lower = 2**depth-1
            upper = min(2**(depth+1)-1, ln+1)
        node = None
        # to pick from known positions
        while node is None:
            rv = r.randrange(max(lower,1), upper)
            node = self.getNode( rv )
        assert(self.root != node)
        return node

    def setModified(self, v):
        self.modified = v

    def isModified(self):
        return self.modified

    def getParent(self, node: Node):
        """
        Get parent of node
        """
        pos = node.getPosition()
        if pos == 0:
            return None
        return self.getNode(pos//2) if pos & 1 else self.getNode(pos//2 -1)

    def logState(self):
        logger.debug("Current state = rnodes {}\n lnodes = {}".format(self.getNodes(), self.nodes))

    def _deleteNode(self, pos):
        del self.nodes[pos]

    def _removeNode(self, node: Node, newnode: Node):
        """
        Remove node from tree, replace with newnode.

        First stage in splicing a subtree, subtree with root node is unlinked, newnode is placed in.
        """
        self.setModified(True)
        self.modifiedDepth=True
        self.testInvariant()
        # Unlink the current node
        npos = node.getPosition()
        self._deleteNode(npos)
        parent = self.getParent(node)
        newnode.setPosition(npos)
        # Replace current node with new node
        if parent:
            if npos & 1:
                assert(parent.getChildren())
                parent.getChildren()[0] = newnode
            else:
                parent.getChildren()[1] = newnode
        self.nodes[npos] = newnode
        if npos == 0:
            self.root=newnode
        # Unlink all children
        cdrn = node.getAllChildren()
        for c in cdrn:
            self._deleteNode(c.getPosition())
            #self.nodes[c.getPosition()]=None


    def spliceSubTree(self, node: Node, newnode: Node):
        """
        Remove subtree with root node, replace it with subtree with root newnode
        """
        assert(node != self.getRoot())
        self._removeNode(node, newnode)
        # newnode is in place, make sure its subtree is updated and children linked in
        newnode.updatePosition()
        nodes = newnode.getChildren()[:]
        while len(nodes) != 0:
            newnodes = []
            for c in nodes:
                newnodes += c.getChildren()[:]
                self._addNode(c, c.getPosition())
            nodes = newnodes
        self.root.clearConstExprCache()
        self.setDataPointCount()
        self.testInvariant()

    def getConstants(self):
        """
        Return an ordered array of constants for all nodes.

        Non initialized constants will have a value of None
        """
        return [ c.getConstant() for c in self.getNodes()]

    def getVariables(self):
        return self.getRoot().getVariables()

    def updateVariables(self, newdataset):
        assert(len(newdataset) and len(newdataset[0]))
        V = self.getVariables()
        for v in V:
            v.setValues(newdataset[v.getIndex()])
        self.setDataPointCount()
        for x in newdataset:
            assert(len(x) == self.getDataPointCount() or self.getDataPointCount() == 0)

    def printNodes(self):
        """
        Print nodes to stdout in binary order (root, ... , ith generation, ....)
        """
        for i, n in enumerate(self.nodes):
            print("Node {} at {}".format(n, i))

    def getRoot(self):
        assert(self.root)
        return self.root

    def getComplexity(self):
        """
        Complexity is defined by the weight of the functions used in the expression.
        """
        return sum(d.getNodeComplexity() for d in self.getNodes())

    def getScaledComplexity(self):
        """
        Return the functional complexity of this tree.

        This a a ratio in [0.0, 1.0] defining how complex the functions are used in this tree.
        """
        c = self.getComplexity()
        d = self.getDepth()
        # a tree of depth d has 2^(d+1) - 1 nodes if full.
        # Discarding leafs, this gives a set of 2^(d)-1 function nodes, with maximum complexity k this
        # results into (2^-1)k
        nodecount = 2**d-1
        maxc = nodecount * Constants.MAX_COMPLEXITY
        # Minimum complexity is with a tree all single arity min complex.
        minc = (d-1) * Constants.MIN_COMPLEXITY
        if c-minc < 0:
            assert(False)
            return 1
        val = ( float(c-minc) / float(maxc-minc))
        assert(val <= 1)
        return val


    @staticmethod
    def swapSubtrees(left, right, depth = None, rng = None, symmetric=True):
        """
        Given two trees, pick random subtree roots and swap them between the trees.

        Will swap out subtrees at equal depth.
        """
        lefttreedepth = left.getDepth()
        righttreedepth = right.getDepth()
        if rng is None:
            logger.warning("Using non deterministic mode")
            rng = getRandom()

        if depth is None:
            if symmetric:
                depthleft = rng.randint(1, min(lefttreedepth, righttreedepth))
                depthright = depthleft
            else:
                depthleft = rng.randint(1, lefttreedepth)
                depthright = rng.randint(1, righttreedepth)
        else:
            if symmetric:
                assert(depth[0]==depth[1])
                depthleft = depth[0]
                depthright = depth[1]
            else:
                depthleft = depth[0]
                depthright = depth[1]

        leftsubroot = left.getRandomNode(seed=None, depth=depthleft, rng=rng)
        rightsubroot = right.getRandomNode(seed=None, depth=depthright, rng=rng)

        leftcopy = copyObject(leftsubroot)
        rightcopy = copyObject(rightsubroot)

        left.spliceSubTree(leftsubroot, rightcopy)
        right.spliceSubTree(rightsubroot, leftcopy)

    @staticmethod
    def createTreeFromExpression(expr: str, variables=None):
        """
        Given an infix expression containing floats, operators, function defined in functions.functionset, create a corresponding expression tree.

        :attention : logarithm is a binary operator : log(x,base), with shorthand ln(x) is allowed but not log(x) with implicit base e
        :bug : 2 * log(3 , 4) is parsed incorrectly
        """
        pfix = infixToPrefix(tokenize(expr, variables))
        result = Tree()
        lastnode = None
        for token in pfix:
            if isFunction(token) or isOperator(token):
                lastnode = result.makeInternalNode(token, lastnode)
            else:
                if isinstance(token, Constant):
                    result.makeConstant(token, lastnode)
                elif isinstance(token, Variable):
                    result.makeLeaf(token, lastnode)
            # If the current node has all children slots filled, move up to parent (until root), else keep completing
            while lastnode.finalized():
                parent = result.getParent(lastnode) # relies on position
                if parent:
                    lastnode = parent
                else:
                    break
        result.setDataPointCount()
        result.testInvariant()
        return result

    def isConstantExpression(self):
        """
        Check if this tree represents a constant expression.
        """
        assert(self.root)
        return self.root.isConstantExpression()

    def getFeatures(self, unique=False):
        """
        Return a sorted integer list of feature indices used in this tree.

        :param bool unique: return unique set. If False, for an expression of the form sin(x3) + cos(x3) will return [3,3], else [3]
        """
        vs = self.getVariables()
        vlist = [v.getIndex() for v in vs]
        if unique:
            vlist = list(set(vlist))
        return sorted(vlist)

    def isConstantExpressionLazy(self):
        return self.root.isConstantExpressionLazy()

    def toExpression(self):
        """
        Print the tree to an infix expression.
        """
        return Node.nodeToExpression(self.root)

    def doConstantFolding(self):
        rootctexpr = self.root.isConstantExpression()
        if rootctexpr:
            value = Node.evaluateAsTree(self.root)
            newroot = ConstantNode(0, Constant(value))
            self.root=newroot
            self.modified = True
            self.modifiedDepth = True
            self.nodes = SortedDict()
            self.nodes[0] = newroot
        else:
            subtrees = flatten(self.root.getConstantSubtrees())
            subtrees = [x for x in subtrees if x is not None]
            values = [Node.evaluateAsTree(x) for x in subtrees]
            for subtreeroot, value in zip(subtrees, values):
                newroot = ConstantNode(subtreeroot.getPosition(), Constant(value))
                self.spliceSubTree(subtreeroot, newroot)
