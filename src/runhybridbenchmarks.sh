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

fases=(2 5)
optimizers=( none de abc pso )
strategies=( 0 )

for expressionid in {0..14}
#do python3 -m gp.paralleldriver -e $expressionid;
do
    for optimizer in "${optimizers[@]}";
    do
        for strategy in "${strategies[@]}"
        do
            for fase in "${fases[@]}";
            do
                #echo -c $processcount -e $expressionid -m $MD -i 4 -t "$topo" -f $fase
                #echo -c 1 -t none -e $expressionid -m $MD -i 4 -j $strategy -k $optimizer -f $fase
                python3 -m gp.paralleldriver -c 1 -t none -e $expressionid -m $MD -i 4 -j $strategy -k $optimizer -f $fase
            done
        done
    done
done
