#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


from tools import msb, compareLists
from functions import functionset, getRandomFunction, tokenize, infixToPostfix, isFunction, isOperator, infixToPrefix
from random import  choice, random
from copy import deepcopy
from node import Node, Constant, ConstantNode, Variable, VariableNode
import logging
import random
import math
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')

class Tree:
    """
        Expression tree. A binary tree stored as an ordered list.
        Internal nodes hold operators, terminal nodes either Variables or Constants.
        Each node has an optional constant.
    """
    def __init__(self):
        self.nodes = []
        self.root = None
        self.variables = {}
        self.evaluated = 0
        self.modified = False
        self.depth = None

    def getNode(self, pos):
        self.testInvariant()
        return self.nodes[pos]

    def testInvariant(self):
        left = self.getNodes()
        right = self.nodes
        middle = self._positionalNodes()
        if not (compareLists(left, right) and compareLists(left, middle)) :
            logger.error("Invariant failed")
            logger.error("getNodes() {}".format(left))
            logger.error("self.nodes {}".format(right))
            raise ValueError("Mismatched lists")
        else:
            for n in left:
                if not n.finalized():
                    logger.error("Node {} has invalid set of children".format(n))
                    raise ValueError("Invalid tree state")

    def _addNode(self, node, pos):
        """
            Add node to tree (without linking), insert variable if a terminal node.
        """
        self.modified = True
        logger.debug("Adding {} at pos {}".format(node, pos))
        if pos == 0:
            self.root = node
        curlen = len(self.nodes)
        if pos >= len(self.nodes):
            self.nodes.extend([None] * (pos-curlen + 1))
            logger.info("Extending list from {} to {}".format(curlen, len(self.nodes)))
        if self.nodes[pos]:
            raise ValueError("Node exists at {}".format(pos))
        else:
            self.nodes[pos] = node
        variable = node.getVariable()
        if variable:
            index = variable.getIndex()
            logger.debug("Adding var {} from node {} to variables".format(variable, node))
            if index in self.variables:
                v = self.variables[index]
                logger.debug("Updating refcount for {} from {} to {}".format(variable, v[1], v[1]+1))
                self.variables[index] = [v[0], v[1]+1]
            else:
                self.variables[index] = [variable, 1]

    def makeInternalNode(self, function, parent=None, constant=None):
        """
            Add an internal node to the tree
        """
        position = 0
        if parent is not None:
            children = parent.getChildren()
            if children:
                position = children[-1].getPosition()+1
            else:
                position = parent.getPosition()*2 + 1
        n = Node(function, position, constant)
        if parent is not None:
            parent.addChild(n)
        else:
            self.root = n
        self._addNode(n, position)
        return n

    def getNodes(self):
        """
            Recursively retrieve all nodes
        """
        result = [self.root]
        result += self.root.getAllChildren()
        return result

    def _positionalNodes(self):
        """
            Get all sorted nodes
        """
        return [d for d in self.nodes if d]

    def isLeaf(self, position):
        """
            Return true if node at position is a terminal/leaf node.
        """
        l = 2*position + 1
        r = 2*position + 2
        if r < len(self.nodes):
            return not (self.nodes[l] or self.nodes[r])
        elif l < len(self.nodes):
            return not self.nodes[l]
        else:
            return True

    def makeLeaf(self, variable, parent, constant = None):
        """
            Make a leaf node holding a reference to a variable.
        """
        logger.debug("Makeleaf with args = v={}, parent={}, cte={}".format(variable, parent, constant))
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

    def makeConstant(self, constant, parent):
        """
            Make a leaf node holding a constant
        """
#        print("Makeleaf with args = v={}, parent={}, cte={}".format(variable, parent, constant))
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
        output = str(root) + "\n"
        children = root.getChildren()[:]
        while len(children) != 0:
            gchildren = []
            for c in children:
                output += str(c) + " "
                gchildren += c.getChildren()[:]
            children = gchildren
            output += "\n"
        return output

    def evaluateTree(self):
        """
            Evaluates tree if tree was modified, else returns a cached results.
            Updates depth if needed.
        """
        if self.modified:
            oe = self.evaluated
            self.evaluated = self._evalTree(self.nodes[0])
            logger.info("Tree state modified, reevaluating from {} to {}".format(oe, self.evaluated))
            self.getDepth()
            self.modified = False
        return self.evaluated

    def _evalTree(self, node):
        """
            Recursively evaluate tree with node as root.
        """
        children = node.getChildren()
        if children:
            value = []
            for child in children:
                v=self._evalTree(child)
                value.append(v)
            try:
                v = node.evaluate(value)
                return v
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                logger.error("Node {} is invalid with given args {}".format(node, value))
                # get current depth, seed, variables
                # start with swapping children
                # Exceptions is too expensive, find semantic correct children
                return 1
        else:
            return node.evaluate()


    def printToDot(self, name = None):
        filename = name or "output.dot"
        handle = open(filename, 'w')
        handle.write("digraph BST{\n")
        for c in self.nodes:
            if c:
                logging.debug("Writing {} to dot".format(c))
                handle.write( str( id(c) ) + "[label = \"" + str(c) + "\"]" "\n")
        for n in self.nodes:
            if n:
                for c in n.getChildren():
                    handle.write(str(id(n)) + " -> " + str(id(c)) + ";\n")
        handle.write("}\n")
        handle.close()

    def getDepth(self):
        """
            Return depth of this tree.
            Returns a cached version or calculates a new one if needed.
        """
        if self.modified:
            self.depth = self.calculateDepth()
        return self.depth

    def calculateDepth(self):
        i = -1
        n = self.nodes[i]
        d = -1
        while n is None:
            i -= 1
            n = self.nodes[i]
        d = n.getDepth()
        return d

    @staticmethod
    def makeRandomTree(variables, depth, seed = None, rng=None, tokenLeafs=False):
        """
            Generate a random expression tree with a random selection of variables
            Topdown construction, there is no guarantee that this construction renders a semantically valid tree
        """
        logger.debug("mkRandomTree with args {} depth {} seed {}".format(variables, depth, seed))
        t = Tree()
        _rng = rng or random.Random()
        if seed is not None: _rng.seed(seed)
        nodes = [t.makeInternalNode(getRandomFunction(rng=_rng), None, None)]
        for i in range(depth):
            # for all generated nodes in last iteration
            newnodes=[]
            for node in nodes:
                for j in range(node.getArity()):
                    if i >= depth-1:
                        if tokenLeafs:
                            child = t.makeConstant(Constant(1.0), node)
                        else:
                            if (_rng.randrange(0, 2) & 1) and variables:
                                child = t.makeLeaf(_rng.choice(variables), node)
                            else:
                                child = t.makeConstant(Constant.generateConstant(rng=_rng), node)
                    else:
                        child = t.makeInternalNode(getRandomFunction(rng=_rng), node, None)
                        newnodes.append(child)
            nodes = newnodes
        return t

    @staticmethod
    def constructFromSubtrees(left, right, seed=None, rng=None):
        logger.debug("cfSubtree with args left {} right {} seed {} rng {} ".format(left, right, seed, rng))
        t = Tree.makeRandomTree(variables=None, depth=1, seed=seed, rng=rng, tokenLeafs=True)
        if t.getRoot().getArity() == 2:
            t.spliceSubTree(t.getNode(1), left.getRoot())
            t.spliceSubTree(t.getNode(2), right.getRoot())
        else:
            t.spliceSubTree(t.getNode(1), left.getRoot())
        return t

    @staticmethod
    def growTree(variables, depth, seed=None):
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        return Tree._growTree(variables, depth, rng=rng)

    @staticmethod
    def _growTree(variables, depth, rng=None):
        if depth == 2:
            left = Tree.makeRandomTree(variables, depth=1, seed=None, rng=rng)
            right = Tree.makeRandomTree(variables, depth=1, seed=None, rng=rng)
            root = Tree.constructFromSubtrees(left, right, seed=None, rng=rng)
            return root
        else:
            left = Tree._growTree(variables, depth-1, rng=rng)
            right = Tree._growTree(variables, depth-1, rng=rng)
            root = Tree.constructFromSubtrees(left, right, seed=None, rng=rng)
            return root


    def getRandomNode(self, seed = None, depth = None):
        """
            Return a randomly selected node from this tree
        """
        logger.debug("Randomnode with seed {}".format(seed))
        r = random.Random()
        if seed is not None:
            r.seed(seed)
        if depth:
            assert(depth < math.log(len(self.nodes)+1, 2))
            lower = 2**depth-1
            upper = min(2**(depth+1)-1, len(self.nodes))
            depthnodes = self.nodes[lower:upper]
            node = None
            while node is None or node is self.getRoot():
                node = r.choice(depthnodes)
            return node
        else:
            node = None
            while node is None or node is self.getRoot():
                node = r.choice(self.nodes)
            return node

    def setModified(self, v):
        self.modified = v

    def isModified(self):
        return self.modified

    def getParent(self, node):
        """
            Get parent of node
        """
        pos = node.getPosition()
        parentindex = 0
        if pos == 0:
            return None
        if pos & 1:
            parentindex = pos//2
        else:
            parentindex = pos//2 -1
        parent = self.nodes[parentindex]
        return parent

    def logState(self):
        logger.debug("Current state = rnodes {}\n lnodes = {}".format(self.getNodes(), self.nodes))

    def removeNode(self, node, newnode):
        """
            Remove node from tree, replace with newnode.
            First stage in splicing a subtree, subtree with root node is unlinked, newnode is placed in.
        """
        self.setModified(True)
        self.testInvariant()
        npos = node.getPosition()
        self.nodes[npos] = None
        parent = self.getParent(node)
        newnode.setPosition(npos)
        if parent:
            if npos & 1:
                logger.debug("Setting newnode as first child of {}".format(parent))
                parent.getChildren()[0] = newnode
            else:
                logger.debug("Setting newnode as second child of {}".format(parent))
                parent.getChildren()[1] = newnode
        self.nodes[npos] = newnode
        if npos == 0:
            self.root=newnode
        # Unlink all children
        cdrn = node.getAllChildren()
        logger.debug("unlinking all children {}".format(cdrn))
        for c in cdrn:
            self.nodes[c.getPosition()]=None
            var = c.getVariable()
            if var:
                logger.debug("Removing var {} from {}".format(var, self.variables))
                logging.debug("Removing var {}".format(var))
                index = var.getIndex()
                if index in self.variables:
                    v = self.variables[index]
                    if v[1] == 1:
                        logger.debug("Removing var {}".format(var))
                        del self.variables[index]
                    else:
                        logger.debug("Decrementing refcount var {} to {}".format(var, v[1]-1))
                        self.variables[index] = [v[0], v[1]-1]



    def spliceSubTree(self, node, newnode):
        """
            Remove subtree with root node, replace it with subtree with root newnode
        """
        logger.debug("Splicing newnode {} in node's {} position".format(newnode, node))
        assert(node != self.getRoot())
        self.removeNode(node, newnode)
        newnode.updatePosition()
        nodes = newnode.getChildren()[:]
        while len(nodes) != 0:
            newnodes = []
            for c in nodes:
                newnodes += c.getChildren()[:]
                logger.debug("Adding child {} at pos {}".format(c, c.getPosition()))
                self._addNode(c, c.getPosition())
            nodes = newnodes
        self.testInvariant()

    def getConstants(self):
        """
            Return an ordered array of constants for all nodes.
            Non initialized constants will have a value of None
        """
        return [ c.getConstant() for c in self.nodes]

    def getVariables(self):
        """
            Return an ordered list of all variables used in this tree
        """
        return self.variables

    def _mergeVariables(self, otherset):
        """
            When a new subtree is merged, update the variables
        """
        variables = self.getVariables()
        logger.debug("Variables is now {}".format(variables))
        for k, v in otherset.items():
            logger.debug("Updating with {} {}".format(k, v))
            if k in variables:
                entry = variables[k]
                entry[1] += v[1]
                variables[k] = entry
            else:
                variables[k] = v
        logger.debug("Variables is now {}".format(variables))
        assert(variables == self.getVariables())

    def _updateVariables(self, vlist):
        """
            When a set of variables from a different tree is merged, update the current set.
        """
        variables = self.getVariables()
        logger.debug("Updating variables is now {}, adding {}".format(variables, vlist))
        for v in vlist:
            k = v.getIndex()
            if k in variables:
                entry = variables[k]
                entry[1] += 1
                variables[k] = entry
            else:
                variables[k] = [v, 1]
        logger.debug("Variables is now {}".format(variables))



    def printNodes(self):
        for i, n in enumerate(self.nodes):
            print("Node {} at {}".format(n, i))

    def updateIndex(self,i=-1):
        """
            Update the variables in the tree s.t. they point at the next datapoint
        """
        # Usually datapoints are present for all items.
        self.setModified(True)
        for k, v in self.variables.items():
            variable = self.variables[k][0]
            index = variable.getCurrentIndex()
            if i != -1:
                variable.setCurrentIndex(i)
            else:
                variable.setCurrentIndex(index + 1)
            self.variables[k][0] = variable
            logger.debug("Updated v = {} index from {} to {}".format(variable, index, variable.getCurrentIndex()))


    def getRoot(self):
        assert(self.root)
        return self.root

    @staticmethod
    def swapSubtrees(left, right, seed = None, depth = None):
        """
            Given two trees, pick random subtree roots and swap them between the trees.
            Will swap out subtrees at equal depth.
        """
        leftsubroot = left.getRandomNode(seed=seed, depth=depth)
        leftv = leftsubroot.getVariables()
        logger.debug("Selected left node {}".format(leftsubroot))

        rightseed = None
        if seed is not None:
            rightseed = seed+42
        rightsubroot = right.getRandomNode(seed=rightseed, depth=depth)
        rightv = rightsubroot.getVariables()
        logger.debug("Selected right node {}".format(rightsubroot))

        leftcopy = deepcopy(leftsubroot)
        rightcopy = deepcopy(rightsubroot)

        left.spliceSubTree(leftsubroot, rightcopy)
        left._updateVariables(rightv)

        right.spliceSubTree(rightsubroot, leftcopy)
        right._updateVariables(leftv)

    @staticmethod
    def createTreeFromExpression(expr, variables=None):
        """
            Given an infix expression containing floats, operators, function defined in functions.functionset,
            create a corresponding expression tree.
        """
        pfix = infixToPrefix(tokenize(expr, variables))
        logger.debug("Create Tree with args \n{} in prefix\n{}".format(expr , pfix))
        result = Tree()
        lastnode = None
        for token in pfix:
            logger.debug("Converting token {} to node".format(token))
            if isFunction(token) or isOperator(token):
                lastnode = result.makeInternalNode(token, lastnode)
            else:
                if isinstance(token, Constant):
                    result.makeConstant(token, lastnode)
                elif isinstance(token, Variable):
                    result.makeLeaf(token, lastnode)
            while lastnode.finalized():
                parent = result.getParent(lastnode) # relies on position
                if parent:
                    lastnode = parent
                else:
                    break
        result.testInvariant()
        return result


    def toExpression(self):
        """
            Print the tree to an infix expression.
        """
        return Node.nodeToExpression(self.root)
