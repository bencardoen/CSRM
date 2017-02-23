#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
import random
from bokeh.plotting import Figure, show, output_file
from bokeh.palettes import Blues9
from gp.algorithm import probabilityMutate as probability

#
## prototype code to visualize mutation behavior with a cooling effect
#
#def probability(generation:int, generations:int, ranking:int, population:int, rng:random.Random=random.Random()):
#    q = (generation / generations) * 0.5
#    w = (ranking / population) * 2
#    r = rng.uniform(a=0,b=1)
#    s = rng.uniform(a=0, b=1)
#    return r > q and s < w
    


def plotValues(values, boolean=True):
    labelx = "Sample sorted in decreasing fitness"
    labely = "Generation"
    p = Figure(webgl=True,title="title", x_axis_label=labelx, y_axis_label=labely)
    xr = len(values[0])
    yr = len(values)
    for x in range(xr):
        for y in range(yr):
            color = 'blue'
            alpha=0.2
            if boolean:
                if values[y][x]:
                    color = 'red'
            else:
                alpha=values[y][x]
            p.circle(x, y, color=color, fill_alpha=alpha, size=10)

    output_file("mutation.html", title="operator probability")
    
    show(p)


if __name__=="__main__":
    trials = 200
    c = [0 for x in range(trials)]
    generations = 15
    population = 15
    rng = random.Random()
    rng.seed(0)
    condensed = [[[0 for k in range(trials)] for r in range(population)] for g in range(generations)]
    for i in range(trials):
        k = 0
        values = [[ probability(g, generations, r, population, rng) for r in range(population) ] for g in range(generations)] 
        for g in range(generations):
            for r in range(population):
                if values[g][r]:
                    k += 1    
                condensed[g][r][i] = values[g][r]
        c[i] = k / (generations*population)
    print(sum(c) / len(c))
    for g in range(generations):
        for r in range(population):
            l = condensed[g][r]
            condensed[g][r] = (sum(l)/len(l))
    plotValues(condensed, False)
#        print(values)
        #print("plotting")
        #plotValues(values)