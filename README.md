# README #

### Dependencies ###
* Python3
* Sortedcontainers : 
    ```Shell
	pip3 install sortedcontainers
    ```
* Optional : graphviz

The project in its current state has code in the subpackage expression holding the datastructures for the GP algorithm. Tree repesentation, expression parsing from/to trees, mutation/crossover operators, random tree initialization, constant/feature management, base functions (functions.py) and several auxillary functions (tools.py).
The executable code in src/expression/testtree.py is an extensive set of tests covering the above functionality

You can execute the tests from the 'src' folder by executing:

```Shell
    $python3 -m expression.testtree
    $python3 -m gp.testalgorithm
```

For most of the tests tree are generated, these are written in dot format in the 'output' folder.
A small script is present in this folder that call graphviz to render the trees in svg format, run from within the output folder:

```Shell
   $./dotify.sh
```

Documentation is present in the folder 'doc/html/index/html'
