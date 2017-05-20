import json
import os
import numpy
from math import log

resultfile = "Collected results for all processes"


def preprocess(dirnames):
    keys = ["datapointcount20_datapointrange(1, 5)", "archivefileNone", "optimizerNone_optimizestrategyNone", "rangesNone", "variablepoint5"]
    for d in dirnames:
        dold = d[:]
        for k in keys:
            d = d.replace(k, "")
        print("\n{}\n is renamed to \n{}\n".format(dold, d))
        os.rename(dold, d)


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


def getCommunicationSize(dirname):
    return getKey(dirname, "communicationsize")


def getTopology(dirname):
    return getKey(dirname, "topo")


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
topologies = ["TreeTopology", "RandomStaticTopology", "VonNeumannTopology"]
expressions = ["expr" + str(i) for i in range(15)]
phases = ["phases" + str(i) for i in [20]]
communicationsizes = [2, 4]
topologiescombined = [t+str(c) for t in topologies for c in communicationsizes]


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

    inv = {a : {m : {} for m in measures} for a in topologiescombined}
    for e, ev in res.items():
        expression = e
        for fk, fv in ev.items():
            measure = fk
            for a, value in fv.items():
                inv[a][measure][expression] = value

    return inv


def writePlotsCSV(res, name):
    #line = algo followed by values for each expression
    with open(name, "w") as f:
        inv = invertresults(res)
        for measure in measures:
            f.write(measure + "\n")
            f.write("topologies, ")
            for i,h in enumerate(expressions):
                f.write(str(i))
                if i != len(expressions):
                    f.write(",")
            f.write("\n")
            for topology in inv:
                if topology == "PassThroughOptimizer":
                    continue
                f.write(topology + ",")
                values = [None ]* len(expressions)
                #print(values)
                for i, expression in enumerate(expressions):
                    vi = inv[topology][measure][expression]
                    if vi:
                        values[i] = -log(vi, 10)
                    else:
                        values[i] = 16


                for j, v in enumerate(values):
                    f.write("{0: 1.3e}".format(v))
                    if j != len(values):
                        f.write(',')
                f.write("\n")





if __name__=="__main__":
    dirs = os.walk('.')
    dirnames = [x[0] for x in dirs if x[0] != "."]
    fullresults = {}
    #preprocess(dirnames)
    for e in expressions:
        print("Parsing {}".format(e))
        v = select(dirnames, [e+"_"])
        assert(len(v) == len(topologiescombined))
        trainingfitness = {}
        meantrainingfitness = {}
        fullfitness = {}
        meanfullfitness = {}
        types = set()
        for q in v:
            t = getTopology(q)
            cs = getCommunicationSize(q)
            assert( t+cs not in types)
            types.add(t+cs)
            tf = getValue(q, "last_fitness")
            ff = getValue(q, "fitness")
            trainingfitness[t+cs] = tf[0]
            assert(min(tf) == tf[0])
            fullfitness[t+cs] = ff[0]
            meantrainingfitness[t+cs] = numpy.mean(tf[0:5])
            meanfullfitness[t+cs] = numpy.mean(ff[0:5])

        fullresults[e] = {"trainingfitness":trainingfitness, "fullfitness":fullfitness, "meantrainingfitness":meantrainingfitness, "meanfullfitness":meanfullfitness}

    #
    #     writeResultsCSV(fullresults, "hybridresults_{}_phases.csv".format(f))
    #     writeResultsJson(fullresults, "hybridresults_{}_phases.json".format(f))
    writePlotsCSV(fullresults, "distributedplots.csv")
