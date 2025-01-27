import json
import os
import numpy
from math import log

resultfile = "Collected results for all processes"


def preprocess(dirnames):
    for d in dirnames:
        #print(len(d))
        t = d.find("expr")
        if t != -1:
            #print("Renaming \n {} to \n {}".format(d, d[t:]))
            os.rename(d, d[t:])


def select(values, filter):
    sel = []
    for v in values:
        for f in filter:
            if f in v:
                sel.append(v)
    return sel


def getOptimizer(dirname):
    return getKey(dirname, "optimizer")


def getKey(dirname, key):
    t = dirname.find(key)
    if t == -1:
        raise ValueError
    p = dirname[t+len(key):].partition("_")[0]
    return p


def getPhases(dirname):
    return getKey(dirname, "phases")


def getValue(dirname, valuekey):
    with open(dirname+ "/" + resultfile) as data:
        jd = json.load(data)
        dictv = jd[0]
        return dictv[valuekey]
    raise ValueError


def writeResultsJson(res, fname):
    with open(fname, 'w') as f:
        json.dump(res, f)


csvheaders = ["expression", "optimizer", "mean training fitness", "mean test fitness"]
measures = ["trainingfitness", "fullfitness", "meantrainingfitness", "meanfullfitness"]
algorithms = ["PSO", "DE", "ABC", "PassThroughOptimizer"]
expressions = ["expr" + str(i) for i in range(15)]
phases = ["phases" + str(i) for i in [2, 5, 10]]


def writeResultsCSV(res, fname):
    with open(fname, "w") as f:
        for i,h in enumerate(csvheaders):
            f.write(h)
            if i != len(csvheaders):
                f.write(",")
        f.write("\n")
        written = set()
        for expr, results in res.items():
            for key, fresults in results.items():
                for algorithm in fresults:
                    r = "{}, {}, {}, {}\n".format(expr, algorithm, results["trainingfitness"][algorithm], results["fullfitness"][algorithm])
                    if r not in written:
                        f.write(r)
                        written.add(r)


def invertresults(res):
    inv = {a : {m : {} for m in measures} for a in algorithms}
    for e, ev in res.items():
        expression = e
        for fk, fv in ev.items():
            measure = fk
            for a, value in fv.items():
                algorithm = a
                inv[algorithm][measure][expression] = value
    return inv


def writePlotsCSV(res, name):
    #line = algo followed by values for each expression
    with open(name, "w") as f:
        inv = invertresults(res)
        for measure in measures:
            f.write(measure + "\n")
            f.write("algorithm, ")
            for i,h in enumerate(expressions):
                f.write(str(i))
                if i != len(expressions):
                    f.write(",")
            f.write("\n")
            for algorithm in inv:
                if algorithm == "PassThroughOptimizer":
                    continue
                f.write(algorithm + ",")
                values = [None ]* len(expressions)
                #print(values)
                for i, expression in enumerate(expressions):
                    vi = inv[algorithm][measure][expression]
                    pi = inv["PassThroughOptimizer"][measure][expression]
                    if vi == 0:
                        if pi == 0:
                            values[i] = 1
                        else:
                            values[i] = -log(pi, 10)

                    else:
                        values[i] = pi / vi
                    #     values[i] = vi
                    # values[i] = pi - vi # 0.2 - 0.1 = 0.1

                for j, v in enumerate(values):
                    f.write("{0: 1.3e}".format(v))
                    if j != len(values):
                        f.write(',')
                f.write("\n")





if __name__=="__main__":
    dirs = os.walk('.')
    dirnames = [x[0] for x in dirs]
    preprocess(dirnames)
    fullresults = {}
    for f in phases:
        for e in expressions:
            v = select(dirnames, [e])
            v = select(v, [f])
            trainingfitness = {}
            meantrainingfitness = {}
            fullfitness = {}
            meanfullfitness = {}
            for q in v:
                op = getOptimizer(q)
                tf = getValue(q, "last_fitness")
                ff = getValue(q, "fitness")
                trainingfitness[op] = tf[0]
                print("balh")
                assert(min(tf) == tf[0])
                fullfitness[op] = ff[0]
                meantrainingfitness[op] = numpy.mean(tf[0:5])
                meanfullfitness[op] = numpy.mean(ff[0:5])

            fullresults[e] = {"trainingfitness":trainingfitness, "fullfitness":fullfitness, "meantrainingfitness":meantrainingfitness, "meanfullfitness":meanfullfitness}

        writeResultsCSV(fullresults, "hybridresults_{}_phases.csv".format(f))
        writeResultsJson(fullresults, "hybridresults_{}_phases.json".format(f))
        writePlotsCSV(fullresults, "hybridplots_{}_phases.csv".format(f))
