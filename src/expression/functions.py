#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from math import log, sin, cos
from expression.tools import matchFloat, matchVariable, approximateMultiple, traceFunction, rootmeansquare, rootmeansquarenormalized, pearson
from expression.node import Constant, Variable
from expression.constants import Constants
import random
import logging
import math
logger = logging.getLogger('global')


# Function objects that can be used in an expression.
# Most screen parameters to avoid expensive (frequent) exceptions.
# For example : div(a/0) is not legal, but catching often generated expression
# is too expensive

def plus(a, b):
    return a+b

def minus(a, b):
    return plus(a, -b)

def multiply(a, b):
    return a*b

def power(a, b):
    if a < 0 or abs(b) > Constants.SIZE_LIMIT:
        return None
    if a == 0 and b<0:
        return None
    try:
        return pow(a,b)
    except OverflowError as e:
        logger.error("Overflow with {} ^ {} and exc {}".format(a,b,e))
        raise e


def division(a, b):
    if b == 0:
        return None
    return a/float(b)

def modulo(a, b):
    b = int(b)
    if b == 0:
        return None
    return int(a) % b

def logarithm(a, b):
    if a <= 0 or b <= 0 or b == 1 or abs(b)>Constants.SIZE_LIMIT:
        return None
    return log(a,b)

def maximum(a, b):
    return max(a,b)

def minimum(a,b):
    return min(a,b)

def sine(a):
    return sin(a)

def cosine(a):
    return cos(a)

def absolute(a):
    return abs(a)

def ln(a):
    return logarithm(a, math.e)

def exponential(a):
    return power(math.e, a)

def square_root(a):
    if a < 0:
        return None
    return math.sqrt(a)

def tangent(a):
    if approximateMultiple(a, math.pi, epsilon=0.001):
        return None
    return math.tan(a)

def tangenth(a):
    return math.tanh(a)

def generateOrderedFunctionTable(fset):
    return sorted(fset.keys(), key= lambda i : i.__name__)


# function : {prettyprint, arity, precedence, associativity, class}
# Default hash is location in memory, which changes, so for deterministic choice, use a one time sort.
functionset = { plus:("+", 2, 2, 'L',1), minus:("-", 2, 2, 'L',1),
                multiply:("*", 2, 3, 'L',2), division:("/", 2, 3, 'L',2),modulo:("%", 2, 3, 'L',2),
                power:("**", 2, 3, 'R',3), square_root:("sqrt", 1,3,'R',3),
                logarithm:("log", 2, 4, 'F',3), maximum:("max", 2, 4, 'F',2), minimum:("min", 2, 4, 'F',2),
                ln:("ln", 1,4,'R',3),exponential:("exp", 1,4,'R',3),
                sine:("sin", 1, 4, 'R',3), cosine:("cos", 1, 4, 'R',3), absolute:("abs",1, 4, 'F',2),
                tangenth:("tanh", 1,4, 'R',4), tangent:("tan",1,4,'R',4)
                }

functions = generateOrderedFunctionTable(functionset)

# Testfunctions from M.F. Korns' paper "A baseline symbolic regression algorithm"
testfunctions = [
                    "1.57 + (24.3*x3)", "0.23+(14.2*((x3+x1)/(3.0*x4)))",
                    "-5.41 + (4.9*(((x3-x0)+(x1/x4))/(3*x4)))", "-2.3 + (0.13*sin(x2))",
                    "3.0 + (2.13 * ln(x4))", "1.3 + (0.13*sqrt(x0))",
                    "213.80940889 - (213.80940889*exp(-0.54723748542*x0))",
                    "6.87+(11*sqrt(7.23*x0*x3*x4))","((sqrt(x0)/ln(x1))*exp(x2)/(x3 ** 2))",
                    "0.81 + 24.3 * ( ( 2.0*x1+3.0*x2 **2) /(4.0*x3**3 + 5.0*x4**4) )",
                    "6.87+ 11* cos(7.23*x0**3)", "2.0 - 2.1 * cos(9.8*x0) * sin(1.3*x4)",
                    "32-3.0*( (tan(x0)/tan(x1) )*( tan(x2) / tan(x3) ))",
                    "22 - 4.2*((cos(x0)-tan(x1))*(tanh(x2)/sin(x3)))",
                    "12.0 - 6.0* tan(x0)/exp(x1) * (ln(x2)-tan(x3) ) "
                ]

tokens = { value[0]: key for key, value in list(functionset.items())}
braces = [',', '(',')']
# A reverse map containing the first letter of each function object
prefixes = { value[0][0]: (key, len(value[0])) for key, value in list(functionset.items())}


def getRandomFunction(seed = None, rng=None):
    _rng = rng or random.Random()
    if seed is not None:
        _rng.seed(seed)
    return _rng.choice(functions)

def getFunctionComplexity(f):
    return functionset[f][4]

def tokenize(expression, variables=None):
    """

    :returns: expression parsed as a list of tokens
    """
    output = []
    i = 0
    # Prepare input
    expression = preParse(expression)
    detect = ['(', ','] + [x for x in list(functionset.keys())] # list of tokens following a unary minus
    while i < len(expression):
        c = expression[i]
        if c == '-':
            if i == 0 or (output and output[-1] in detect):
                expression, i, output = handleUnaryMinus(expression, i, output, variables)
            else:
                output.append(tokens['-'])
        elif c == '^':
            output.append(tokens['**'])
        elif c in braces:
            output.append(c)
        elif c in prefixes : # functions
            expression, i, output = parseFunction(expression, i, output)
        else:
            f = matchFloat(expression[i:])
            if f:
                i += len(f)-1  # -1 since we will increment i in any case
                output += [Constant(float(f))]
            else:
                v, i = parseVariable(expression[i:], variables, i)
                if v:
                    output += [v]
                else:
                    logging.error("Invalid pattern in string {} , full expr {}".format(expression[i:], expression))
                    raise ValueError("Failed to decode number, string is {}  in expr{}".format(expression[i:], expression))
        i += 1
    return output


def isFunction(token):
    return token in functionset and functionset[token][3] == 'F'


def isOperator(token):
    return token in functionset and functionset[token][3] != 'F'


def infixToPostfix(infix):
    """
    Dijkstra's shunting yard algorithm to convert an infix expression to postfix
    """
    result = []
    stack = []
    for token in infix:
        if token == ',':
            while stack[-1] != '(':
                result.append(stack.pop(-1))
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                logger.debug("apppending t {}".format(stack[-1]))
                result.append(stack.pop(-1))
            stack.pop(-1)
            if stack and isFunction(stack[-1]):
                logger.debug("apppending f {}".format(stack[-1]))
                result.append(stack.pop(-1))
        elif isinstance(token, Constant):
            result.append(token)
            logger.debug("apppending c {}".format(token))
        elif isinstance(token, Variable):
            result.append(token)
            logger.debug("appending v {}".format(token))
        elif token in functionset:
            f = functionset[token]
            if isFunction(token):
                stack.append(token)
            else:
                oper = functionset[token]
                while True and stack:
                    top = stack[-1]
                    logger.debug("checking for operator precedence {}".format(top))
                    if isOperator(top):
                        operonstack = functionset[top]
                        if oper[3] == 'L' and oper[2] <= operonstack[2]:   # l assoc and less precedence
                            logger.debug(" R and leq : apppending o {}".format(stack[-1]))
                            result.append(stack.pop(-1))
                        elif oper[2] < operonstack[2]:
                            logger.debug(" L and le : apppending o {}".format(stack[-1]))
                            result.append(stack.pop(-1))
                        else:
                            break
                    else:
                        break
                logger.debug("pushing operator {}".format(token))
                stack.append(token)
    while stack:
        logger.debug("apppending last of stack {}".format(stack[-1]))
        result.append(stack.pop(-1))
    return result

def infixToPrefix(infix):
    """
    Convert infix to prefix by way of postfix conversion.
    """
    # Algorithm is mirror of i to p, roles of () are reversed.
    # Reverse input, swap (), i to p, reverse result
    infix = [x for x in reversed(infix)]
    for index,item in enumerate(infix):
        if item == ')':
            infix[index] = '('
        elif item == '(':
            infix[index] = ')'
    tmp = infixToPostfix(infix)
    result = [x for x in reversed(tmp)]
    return result

def parseVariable(stream, variables, index):
    """

    :returns: Variable or None , newindex
    """
    v = matchVariable(stream)
    if v:
        index += len(v)-1
        vindex = int(''.join([x for x in v if x not in ['X', 'x', '_']]))
        return ( Variable(variables[vindex], vindex) , index)
    else:
        return (None, index)

def handleUnaryMinus(expression, index, output, variables):
    f = matchFloat(expression[index:])
    if f:
        index += len(f)-1  # -1 since we will increment i in any case
        output += [Constant(float(f))]
    else:
        v, newindex = parseVariable(expression[index+1:], variables, index)
        if v:
            index = newindex+1
            output += ['(',Constant(-1.0), tokens['*'], v, ')']
        else:
            if index < (len(expression)-1) and expression[index+1] == '(': # case : -(5+3) -> (-1*(5+3))
                lcount = 0
                j = index
                while j != len(expression):
                    if expression[j] == '(':
                        lcount+=1
                    elif expression[j] == ')':
                        lcount -= 1
                        if lcount == 0:
                            if j == len(expression)-1:
                                expression = expression[:] + ')'
                            else:
                                expression = expression[0:j] + ')' + expression[j+1:]
                            break
                    j+=1
                output += ['(',Constant(-1.0), tokens['*']]
            else:
                logging.error("Invalid pattern in string {} , full expr {}".format(expression[index:], expression))
                raise ValueError("Failed to decode number.")
    return expression, index, output


def parseFunction(expression: str, index: int, output: list):
    """
    Decode a function from the stream

    :returns: expression, newindex, newoutput
    """
    # reversed since we want a greedy match, e.g. tanh is preferred over tan|h
    for length in reversed(range(1, 5)):
        name = expression[index:index+length]
        if name in tokens:
            logger.debug("{} at index {}".format(name, index))
            output += [tokens[name]]
            index += length-1
            break
    else:
        raise ValueError("Invalid chars {}".format(name))
    return expression, index, output

def preParse(expression:str):
    """
        Clean up expression before tokenizer operates on it.
    """
    expression = expression.replace(' ', '')
    expression = expression.replace("**", "^")
    return expression

def rmsfitness(actual, expected, tree):
    return fitnessfunction(actual, expected, tree, distancefunction=rootmeansquare)

def rmsnormfitness(actual, expected, tree):
    return fitnessfunction(actual, expected, tree, distancefunction=rootmeansquarenormalized)

def pearsonfitness(actual, expected, tree):
    return fitnessfunction(actual, expected, tree, distancefunction=pearson)

def fitnessfunction(actual, expected, tree, distancefunction=None):
    """
        Fitness function based on a distance measure.
        :param list actual: Y', actual approximated valus
        :param list expected: Y, expected values
        :param Tree tree: the instance to operate on
        :param function distancefunction: function object that accepts actual, expected as parameters and returns a numerical distance. If None, rootmeansquare is used.

        Returns d(Y', Y).
    """
    if not actual:
        logger.debug("Tree instance has no datapoints : invalid")
        return Constants.MINFITNESS
    if len(actual) != len(expected):
        logger.debug("Tree instance has no matching datapoints : invalid")
        return Constants.MINFITNESS

    for j, i in enumerate(actual):
        if i is None:
            logger.debug("Tree instance has an invalid expression for a datapoint {}".format(j))
            return Constants.MINFITNESS

    distance = float('inf')
    if distancefunction is None:
        distance = rootmeansquare(actual, expected)
    else:
        distance = distancefunction(actual, expected)
    return distance
