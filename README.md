# README #

This document gets you up and running in the shortest

### Dependencies ###
* Python3
* Sortedcontainers :

    ```
	$pip3 install sortedcontainers
    ```

* Numpy :

    ```
    $pip3 install numpy
    ```

* Bokeh :

    ```
    $pip3 install bokeh
    ```
* MPI and mpi4py:
    MPI is the C/C++ framework, mpi4py are the required python bindings. The following link will get you started:

    https://pypi.python.org/pypi/mpi4py

    On Ubuntu a typical install would be
    ```
    $sudo apt-get mpich python3-mpi4py
    ```

* Optional : graphviz


### Tests

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

### Source code documentation

Documentation is present in the folder 'doc/html/index/html'

With all that out of the way, let's get started with running the code

## Hello World in Symbolic regression

In a typical use case you have a set of n features, each with k datapoints forming a X = n x k matrix.

You also have a set of expected data, Y, an n x 1 vector.

You want CSRM to find a mathematical expression (non linear) f(X) = Y' s.t. the distance between Y and Y' is minimal.

#### Input Data format
You'll need to encode X and Y in a format that CSRM understands.

CSRM expects a simple CSV textfile, where ',' is the delimiter, and each line is interpreted as k datapoints for a single feature. Values should be in a floating point format (integers are obviously fine).

#### Output
CSRM produces a _lot_ of output, depending on your preferences.
We distinguish between convergence characteristics (i.e. rate of convergence, accuracy, depth, and more..) and the actual results.

CSRM not only offers you the results, but also how it came by them, so don't expect a black box. You can obviously ignore all that data and simply focus on the results.

##### Output format
Output is written to a local folder ./output/<configstring>/
configstring is a concatenation of all configuration options you used in starting the program.
CSRM writes a JSON and HTML file per process. In addition a summarized file (in JSON and HTML) is written that summarizes the all results. Finally, the results of the search, the expressions you wanted to obtain, are written in a text file (bestresults.txt) and a HTML page (bestresults.html).

##### Running
I will give you two copy-pasteable commands that 'just work'. The first is a sequential run, the second a distributed run.

* Sequential

```Python
$python3 -m gp.paralleldriver -x doe/input.csv10 -q 3 -y doe/output.csv10 -d 10 -c 1 -f 20 -p 20 -g 20 -v
```


-x is the input file (3 x 10)

-q specifies the number of features CSRM should expect (3)

-y is the expected data file (1 x 10)

-d is the number of datapoints (10)

-f is the number of phases (restarts reusing the best expressions from the last run)

-p is the population

-g is the number of generations

-c is the number of processes, 1 in this case.

-v instructs CSRM to open all HTML output in your OS's preferred web browser.

Initially you'll be most interested in the 'bestresults.html' page, which lists the best expressions ordered by best fitness score to worst, gives a some statistics about the distribution and lists the configuration you used.

* Distributed
```Python
$mpiexec -n 3 python3 -m gp.paralleldriver -x doe/input.csv10 -q 3 -y doe/output.csv10 -d 10 -c 3 -f 20 -p 20 -g 20 -v -t tree
```
Note that we use MPI now to prefix the command string, and specify both the number of processes and the topology

-n <processcount> Has to match -c <>
-t <topology> The topology to use. Use either tree, grid, none or random.

The results are again written to file and displayed in a browser of your choice.

##### More
-h will give you all parameters and their meaning, with even more information in the documentation.
