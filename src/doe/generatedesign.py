#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


def generate(points, minimum, maximum):
    diff = maximum - minimum
    stepsize = diff / points
    result = [minimum]
    for i in range(points-1):
        point = result[-1] + stepsize
        result.append(point)
    assert(len(result) == points)
    return result


def readSpec(filename, parameters):
    values = []
    with open(filename, 'r') as f:
        contents = f.readlines()
        for line in contents:
            v = line.split()
            if len(v) != parameters:
                continue
            parsed = []
            failedparsing = False
            for sd in v:
                try:
                    parse = int(sd)
                    parsed.append(parse)
                except ValueError as e:
                    failedparsing = True
            if not failedparsing:
                values.append(parsed)

        return values


def mergeDesign(designpoints:list, selection:list, parameters:int):
    assert(len(designpoints))
    result = []
    assert(isinstance(designpoints, list))
    assert(isinstance(selection, list))

    for s in selection:
        assert(len(s) == parameters)
        values = []
        for i, index in enumerate(s):
            values.append(designpoints[i][index])
        result.append(values)
    return result


def writeDesign(design, parameters, filename):
    with open(filename, 'w') as f:
        for i,p in enumerate(parameters):
            f.write(p)
            if i != len(parameters)-1:
                f.write(", ")
            else:
                f.write("\n")
        for dline in design:
            for i, d in enumerate(dline):
                f.write("{}".format(d))
                if i != len(dline)-1:
                    f.write(", ")
                else:
                    f.write("\n")


def reverseMapping(mapping):
    result = [[] for _ in range(len(mapping))]
    for k,v in mapping.items():
        result[v] = k
    return result


parameters = {"R0":(12,20), "Seed":(1,500), "Immunity":(0.75, 0.95)}
paramtoindex = {"R0":0, "Seed":1,"Immunity":2}
indextoparam = reverseMapping(paramtoindex)



if __name__=="__main__":
    points = 30
    results = {}
    for param, rng in parameters.items():
        results[param] = generate(points, rng[0], rng[1])
    orderedvalues = [[], [], []]
    for key,value in results.items():
        orderedvalues[paramtoindex[key]]=(value)
    design = readSpec("Design30.spec", len(parameters))
    completed = mergeDesign(orderedvalues, design, len(parameters))
    writeDesign(completed, indextoparam, "design.txt")
