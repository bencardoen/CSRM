#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from math import log, sin, cos, sqrt
from random import randint, choice
import tree
import re
import logging
import tools
logger = logging.getLogger('global')


## Functions defined by wrapper, allows easier future adaption.
def plus(a, b):
    return a+b

def minus(a, b):
    return plus(a, -b)

def multiply(a, b):
    return a*b

def power(a, b):
    return pow(a,b)

def division(a, b):
    return a/float(b)

def modulo(a, b):
    return a % b

def logarithm(a, b):
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


# function : {prettyprint, arity, precedence, associativity}
functionset = { plus:("+", 2, 2, 'L'), minus:("-", 2, 2, 'L'),
                multiply:("*", 2, 3, 'L'), division:("/", 2, 3, 'L'),modulo:("%", 2, 3, 'L'),
                power:("**", 2, 3, 'R'),
                logarithm:("log", 2, 4, 'F'), maximum:("max", 2, 4, 'F'), minimum:("min", 2, 4, 'F'),
                sine:("sin", 1, 4, 'F'), cosine:("cos", 1, 4, 'F'), absolute:("abs",1, 4, 'F')
                }
# todo complete
testfunctions = [ "1.57 + (24.3*x3)", "0.23+(14.2*((x3+x1)/(3.0*x4)))"
                ]

tokens = { value[0]: key for key, value in list(functionset.items())}

def getRandomFunction():
    return choice(list(functionset.keys()))

def tokenize(expression, variables=None):
    """
        Split expression into tokens, returned as a list.
    """
    output = []
    i = 0
    expression = expression.replace(' ', '')
    detect = ['(', ','] + [x for x in list(functionset.keys())]
    logger.debug("tokenize with args expr {} vars {}".format(expression, variables))
    while i < len(expression):
        c = expression[i]
        logger.debug("Tokenizing {} at index {}".format(c, i))
        if c == '-':
            if i == 0 or (output and output[-1] in detect):
                f = tools.matchFloat(expression[i:])
                if f:
                    i += len(f)-1  # -1 since we will increment i in any case
                    output += [tree.Constant(float(f))]
                else:
                    v, newindex = parseVariable(expression[i+1:], variables, i)
                    if v:
                        i = newindex+1
                        output += ['(',tree.Constant(-1.0), tokens['*'], v, ')']
                    else:
                        if i < (len(expression)-1) and expression[i+1] == '(': # case : -(5+3)
                            lcount = 0
                            j = i
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
                            output += ['(',tree.Constant(-1.0), tokens['*']]
                        else:
                            logging.error("Invalid pattern in string {} , full expr {}".format(expression[i:], expression))
                            raise ValueError("Failed to decode number.")
            else:
                output += [tokens['-']]
        elif c == '^':
            output += [tokens['**']]
            logger.debug("Power at index {}".format(c, i))
        elif c == '*':
            if expression[i:i+2] == '**':
                output += [tokens['**']]
                logger.debug("Power at index {}".format(c, i))
                i += 1
            else:
                logger.debug("Mult at index {}".format(c, i))
                output += [tokens['*']]
        elif c == '(':
            logger.debug("( at index {}".format(c, i))
            output += ['(']
        elif c == ')':
            logger.debug(") at index {}".format(c, i))
            output += [')']
        # match min, max, abs, cos, sin, log
        elif c in ['m', 'a', 'c', 's', 'l'] :
            logger.debug("f at index {}".format(c, i))
            if i != len(expression)-2:
                name = expression[i:i+3]
                if name in tokens:
                    logger.debug("{} at index {}".format(name, i))
                    output += [tokens[name]]
                else:
                    raise ValueError("Invalid chars {}".format(name))
                i += 2
            else:
                raise ValueError("Invalid chars {}".format(name))
        elif c in tokens:
            output += [tokens[c]]
        elif c == ',':
            output += [',']
        else:
            f = tools.matchFloat(expression[i:])
            if f:
                i += len(f)-1  # -1 since we will increment i in any case
                output += [tree.Constant(float(f))]
            else:
                v, i = parseVariable(expression[i:], variables, i)
                if v:
                    output += [v]
                else:
                    logging.error("Invalid pattern in string {} , full expr {}".format(expression[i:], expression))
                    raise ValueError("Failed to decode number.")
        i += 1
    return output

def isFunction(token):
    decision = False
    if token in functionset:
        if functionset[token][3] == 'F':
            decision = True
    logger.debug("Token {} = function ? {}".format(token, decision))
    return decision

def isOperator(token):
    decision = False
    if token in functionset:
        if functionset[token][3] != 'F':
            decision = True
    logger.debug("Token {} = operator ? {}".format(token, decision))
    return decision

def infixToPostfix(infix):
    """
        Dijkstra's shunting yard algorithm to convert an infix expression to postfix
    """
    result = []
    stack = []
    logger.debug("infix to postfix with args \n{}".format(infix))
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
        elif isinstance(token, tree.Constant):
            result.append(token)
            logger.debug("apppending c {}".format(token))
        elif isinstance(token, tree.Variable):
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
    v = tools.matchVariable(stream)
    if v:
        index += len(v)-1
        vindex = int(''.join([x for x in v if x not in ['X', 'x', '_']]))
        vs = None
        vs = variables[vindex]
        variable = tree.Variable(vs, vindex)
        logging.debug("Parsed Variable v={}".format(variable))
        return (variable, index)
    else:
        return (None, index)
