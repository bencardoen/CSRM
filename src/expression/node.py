#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


import logging
import random
import expression.functions
from expression.tools import traceFunction

logger = logging.getLogger('global')

class Node:
    def __init__(self, fun, pos, constant=None):
        """
        Create a node composed of a function object, position in (binary) tree and an optional
        constant multiplier. Arity of the function is assumed to be 1.
        """
        self.function = fun
        self.pos = pos
        self.children = []
        self.constant = constant
        if self.function:
            self.arity = expression.functions.functionset[self.function][1]
        else:
            self.arity = 1
        self._modified = True
        self._cachedeval = 0
        self._depth = Node.positionToDepth(pos)

    @staticmethod
    def positionToDepth(pos):
        i = 0
        while pos:
            if pos & 2:
                pos -= 1
            pos >>= 1
            i+=1
        return i

    def getDepth(self):
        return self._depth

    def getNodeComplexity(self):
        if self.function:
            logger.debug("Retrieving function complexity for {}".format(self.function))
            return expression.functions.getFunctionComplexity(self.function)
        return 0

    @traceFunction
    def evaluate(self, args=None):
        """
        Evaluate this node using the function object, optionally multiplying with the constant.
        """
        logger.debug("Evaluating {} with args {}".format(self, args))
        rv = 0
        if(args and len(args) != self.arity):
            raise  ValueError("Incompatible parity {} and arguments {}".format(self.arity, args))
        if args is None:
            rv = self.function()
        else:
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

    def addChild(self, node):
        if self.arity <= len(self.children):
            raise ValueError("Arity = {} for children {}".format(self.arity, self.children))
        self.children.append(node)

    def getChildren(self):
        return self.children

    def getConstant(self):
        return self.constant

    def getArity(self):
        return self.arity

    def getFunction(self):
        return self.function

    def __str__(self):
        output = "{}".format(expression.functions.functionset[self.function][0])
        if self.constant:
            output += "*{}".format(self.constant.getValue())
        return output

    def __repr__(self):
        fname = "leaf"
        if self.function:
            fname = self.function.__name__
        return " f={} arity={} constant={}".format(fname, self.arity, self.constant or 1)

    @traceFunction
    def updatePosition(self):
        pos = self.getPosition()
        i = 1
        for c in self.getChildren():
            logger.debug("Setting position of {} from {} to {}".format(c, c.getPosition(), 2*pos + i))
            c.setPosition(2*pos + i)
            i+=1
            c.updatePosition()

    @traceFunction
    def getAllChildren(self):
        children = self.getChildren()[:]
        collected = children[:]
        if children :
            for c in children:
                collected += c.getAllChildren()[:]
            logger.debug("Returning collected {}".format(collected))
            return collected
        else:
            return []

    @staticmethod
    @traceFunction
    def nodeToExpression(node):
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
#            return str(node.evaluate())


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
    CONSTANTLOWER = 0
    CONSTANTUPPER = 1
    def __init__(self, value):
        self._value = value

    def getValue(self):
        return self._value

    def setValue(self, nvalue):
        self._value = nvalue

    def __str__(self):
        return "Constant c = {}".format(self._value)

    def __repr__(self):
        return "Constant c = {}".format(self._value)

    @staticmethod
    def generateConstant(seed=None, rng = None):
        """
            Generate a Constant object with value [lower, upper) by randomgenerator
        """
        _rng = rng or random.Random()
        if seed is not None: _rng.seed(seed)
        return Constant(Constant.CONSTANTLOWER + _rng.random()*(Constant.CONSTANTUPPER-Constant.CONSTANTLOWER))


class Variable():
    @staticmethod
    def toVariables(lst:list):
        """
            Converts a twodimensional array where each row is a set of data points to a list of Variables.
        """
        return [Variable(row, i) for i,entry in enumerate(lst)]


    def __init__(self, values, index):
        """
            Values is a list of datapoints this feature has
            Index is the name, or identifier of this feature. E.g "x_09" : index = 9
        """
        self._values = values
        self._index = index
        self._current = 0

    def getValues(self):
        return self._values

    def __len__(self):
        return len(self._values)

    def setValues(self,vals):
        self._values = vals

    def setCurrentIndex(self,i):
        assert(i < len(self._values))
        self._current = i

    def getCurrentIndex(self):
        return self._current

    def getValue(self):
        if self._values:
            return self._values[self._current]
        else:
            logger.error("Variable value not set : returning 1")
            raise ValueError("Variable value not set!!")
            return 1

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

    def evaluate(self, args=None):
        con = self.getConstant()
        if con :
            return self.variable.getValue() * con.getValue()
        else:
            return self.variable.getValue()

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
            output += " * {}".format(self.constant.getValue())
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

    def evaluate(self, args=None):
        return self.getConstant().getValue()

    def getArity(self):
        return 0

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
