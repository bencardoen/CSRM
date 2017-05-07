#!/bin/bash

#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

# Simple script converts all .dot files in cwd to svg
DIR="output"
#cd $DIR
for f in *.dot; do dot -Tsvg -o ${f%.dot}.svg $f; done
