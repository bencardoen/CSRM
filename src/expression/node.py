#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


import logging
import random
from functools import reduce
import expression.functions
from expression.constants import Constants
from expression.tools import traceFunction, getRandom

logger = logging.getLogger('global')


class Node:
    def __init__(self, fun, pos, constant=None):
        """
        Create a node composed of a function object, position in (binary) tree and an optional constant multiplier.

        Arity of the function is assumed to be 1.
        """
        self.function = fun
        self.pos = pos
        self.children = []
        self.constant = constant
        if self.function:
            self.arity = expression.functions.functionset[self.function][1]
        else:
            self.arity = 0
        self._depth = Node.positionToDepth(pos)
        self.ctexpr = None

    def clearConstExprCache(self):
        self.ctexpr = None
        for c in self.children:
            c.clearConstExprCache()

    @staticmethod
    def evaluateAsTree(node, index=None):
        """
        Recursively evaluate tree with node as root.

        Returns None if evaluation is not valid
        """
        children = node.children
        if children:
            arity = node.arity
            value = [None] * arity
            for i, child in enumerate(children):
                v=Node.evaluateAsTree(child, index)
                if v is None:
                    return None
                value[i] = v
            return node.evaluate(value, index=index) # function or operator
        else:
            return node.evaluate(index=index) # leaf

    @staticmethod
    def positionToDepth(pos):
        """
        Use the structure of the binary tree and its list encoding to retrieve depth in log(N) steps without relying on a parent pointer.
        """
        i = 0
        while pos:
            if pos & 2:
                pos -= 1
            pos >>= 1
            i+=1
        return i

    def getDepth(self):
        return self._depth

    def getConstantSubtrees(self):
        """
        Returns the roots of all subtrees that represent a constant expression.

        Requires that isConstantExpression has been called on the tree at least once.
        """
        if not self.isLeaf():
            if self.ctexpr is None:
                logger.info("ctexpr is None for {}".format(self))
            assert(self.ctexpr is not None)
        if self.ctexpr:
            return self
        else:
            if self.children:
                return [c.getConstantSubtrees() for c in self.children]
            else:
                return None

    def isConstantExpression(self):
        """
        Determine if the expression tree with this node as root forms a constant expression.

        Traverses entire tree and sets the attribute ctexpr with the result of this function. This allows for caching, at a cost for large non const expr trees.
        E.g. x2 / (3 + 4) is not constexpr, but to later on efficiently extract subtrees we need full traversal.
        """
        if self.ctexpr is None:
            result = True
            if self.children:
                for c in self.children:
                    if not c.isConstantExpression():
                        result = False
                        # do not break, need full traversal
            else:
                assert(False)
            self.ctexpr = result
        assert(self.ctexpr is not None)
        return self.ctexpr

    def isConstantExpressionLazy(self):
        """
        Lazy version of isConstantExpression, without full traversal.

        For a large non const tree, and without the need for further traversal (e.g. subtrees) this can be far more efficiently.
        E.g. x2 * (some huge const expr tree) returns False with 1 recursive call.
        Does not cache result to avoid clashes with getConstantSubtrees
        """
        if self.children:
            for c in self.children:
                if not c.isConstantExpression():
                    return False
        else:
            assert(False)
        return True

    def getNodeComplexity(self):
        if self.function:
            return expression.functions.getFunctionComplexity(self.function)
        return 0

    def evaluate(self, args=None, index:int=None):
        """
        Evaluate this node using the function object, optionally multiplying with the constant.
        """
        assert(args and len(args) == self.arity)
        rv = self.function(*args)
        if self.constant:
            rv*=self.constant.getValue()
        return rv

    def finalized(self):
        """
        Return true if this node can be evaluated (e.g. has enough children for its arity)
        """
        return ( len(self.getChildren()) == self.getArity())

    def getPosition(self):
        return self.pos

    def setPosition(self, newpos):
        self.pos = newpos
        self._depth = Node.positionToDepth(self.pos)

    def recursiveDepth(self):
        if self.children:
            return 1 + max([node.recursiveDepth() for node in self.children])
        else:
            return 0

    def addChild(self, node):
        if self.arity <= len(self.children):
            raise ValueError("Arity = {} for children {}".format(self.arity, self.children))
        self.children.append(node)

    def getChildren(self):
        return self.children

    def isLeaf(self):
        return False

    def getConstant(self):
        return self.constant

    def getArity(self):
        return self.arity

    def getFunction(self):
        return self.function

    def __str__(self):
        output = "{}".format(expression.functions.functionset[self.function][0])
        if self.constant:
            output = "{}".format(self.constant.getValue()) + " * " + output
        return output

    def __repr__(self):
        fname = "leaf"
        if self.function:
            fname = self.function.__name__
        return " f={} arity={} constant={}".format(fname, self.arity, self.constant or 1)

    def updatePosition(self):
        """
        After operations on a tree, the position of this node can be invalidated.

        After a valid call to setPosition(pos), update all children recursively to correct position.
        """
        pos = self.getPosition()
        i = 1
        for c in self.getChildren():
            c.setPosition(2*pos + i)
            i+=1
            c.updatePosition()

    def getAllChildren(self):
        """
        Returns all descendant nodes.
        """
        children = self.getChildren()[:]
        collected = children[:]
        if children :
            for c in children:
                collected += c.getAllChildren()[:]
            return collected
        else:
            return []

    @staticmethod
    def nodeToExpression(node):
        """
        Top down traversal constructing an arithmetic expression from the tree with this node as root.

        Expression is in infix, so explicit usage of brackets.
        """
        children = node.getChildren()
        arity = node.getArity()
        if children:
            left = Node.nodeToExpression(children[0])
            frepr = expression.functions.functionset[node.getFunction()][0]
            isfunction = expression.functions.functionset[node.getFunction()][3] == 'F'
            if arity == 2:
                right = Node.nodeToExpression(children[1])
                if isfunction:
                    return "( {}( {}, {} ) )".format(frepr, left, right)
                else:
                    return "( {} {} {} )".format(left, frepr, right)
            else :
                return "( {}( {} ) )".format(frepr, left)
        else:
            return str(node.expr())


    def __hash__(self):
        """Despite docs, this isn't always the default implementation."""
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def getVariable(self):
        return None

    def getVariables(self):
        ac = self.getAllChildren()
        vs = []
        for c in ac:
            v = c.getVariable()
            if v:
                vs.append(v)
        return vs

    def expr(self):
        raise NotImplementedError


class Constant():

    def __init__(self, value):
        self._value = value

    def getValue(self):
        return self._value

    def setValue(self, nvalue):
        self._value = nvalue

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def __str__(self):
        return "Constant c = {}".format(self._value)

    def __repr__(self):
        return "Constant c = {}".format(self._value)

    @staticmethod
    def generateConstant(seed=None, rng = None):
        """
        Generate a Constant object with value [lower, upper) by randomgenerator
        """
        _rng = rng
        if rng is None:
            _rng = getRandom()
            if seed is None:
                logger.warning("Non deterministic mode")
        if seed is not None:
            _rng.seed(seed)
        return Constant(Constants.CONSTANTS_LOWER + _rng.random()*(Constants.CONSTANTS_UPPER-Constants.CONSTANTS_LOWER))


class Variable():
    @staticmethod
    def toVariables(lst:list):
        """
        Converts a twodimensional array where each row is a set of data points to a list of Variables.
        """
        return [Variable(entry, i) for i,entry in enumerate(lst)]


    def __init__(self, values, index):
        """
        Values is a list of datapoints this feature has.

        Index is the name, or identifier of this feature. E.g "x_09" : index = 9
        """
        self._values = values
        self._index = index

    def getValues(self):
        return self._values

    def __len__(self):
        return len(self._values)

    def setValues(self,vals):
        self._values = vals

    def getValue(self, index=None):
        assert(self._values)
        if index is None:
            index = 0
        return self._values[index]

    def getIndex(self):
        """
        Return the unique identifier of this feature, NOT the current datapoint
        """
        return self._index

    def __str__(self):
        return "x_{}={}".format(self.getIndex(),self.getValue())

    def __repr__(self):
        return "x_{}={}".format(self.getIndex(),self.getValue())


class VariableNode(Node):
    """
    Represents a leaf or terminal node in the expression, holding a reference to a variable (x_i)
    """

    def __init__(self, pos, variable, constant=None):
        Node.__init__(self, None, pos, constant)
        self.variable = variable

    def getVariable(self):
        return self.variable

    def evaluate(self, args=None, index:int=None):
        con = self.getConstant()
        v = self.variable.getValue(index)
        if con :
            return v * con.getValue()
        else:
            return v

    def isLeaf(self):
        return True

    def isConstantExpression(self):
        return False

    def getArity(self):
        return 0

    def __str__(self):
        output = "{}".format(self.variable)
        if self.constant:
            output += " * {}".format(self.constant.getValue())
        return output

    def __repr__(self):
        output = "{}".format(self.variable)
        if self.constant:
            output = "{}".format(self.constant.getValue()) + " * " + output
        return output

    def __hash__(self):
        #despite docs, this isn't always the default implementation
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def expr(self):
        return "x{}".format(self.getVariable().getIndex())


class ConstantNode(Node):
    """
    Represents a leaf or terminal node in the expression, holding a constant
    """

    def __init__(self, pos, constant):
        Node.__init__(self, None, pos, constant)

    def evaluate(self, args=None, index:int=None):
        return self.getConstant().getValue()

    def getArity(self):
        return 0

    def isConstantExpression(self):
        return True

    def isLeaf(self):
        return True

    def __str__(self):
        return "{}".format(self.getConstant().getValue())

    def __repr__(self):
        return "{}".format(self.getConstant().getValue())

    def __hash__(self):
        #despite docs, this isn't always the default implementation
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def expr(self):
        return self.evaluate()
