#!/bin/bash

#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
MD=10

#mpiexec -n 3 python3 -m gp.paralleldriver -c 3 -t tree

topos=( randomstatic grid tree )
fases=(30)
processcounts=(3 7 15 31)

for expressionid in {0..14}
#do python3 -m gp.paralleldriver -e $expressionid;
do
    for processcount in "${processcounts[@]}";
    do
        for topo in "${topos[@]}"
        do
            for fase in "${fases[@]}";
            do
                mpiexec -n $processcount python3 -m gp.paralleldriver -c $processcount -e $expressionid -m $MD -i 4 -t "$topo" -f $fase
            done
        done
    done
done
