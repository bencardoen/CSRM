# Julia notes

## Requirements

* Julia
* Jupyter
* IJulia (Pkg.add("IJulia"))

## Running
### Console
```bash
$ julia```
### Notebook
```bash
$ jupyter notebook
```
Inside notebook, shift+enter executes, esc switches to command mode. Tab autocompletes.

In command mode, typing m switches code to markdown

Shell commands can be executed by
```Julia
julia>;netstat -tulpna
```


#### Basics
##### Package management
```Julia
julia> Pkg.add("pname")
julia> Pkg.update()
julia> using pname # eq to from x import *
julia> import <packagename>: <name>
julia> include("filename") # executes script in filename
```
##### Help
```Julia
julia> ?<name>
```

##### Introspection
```
julia> typeof(obj)
```
##### Arrays
```
julia> m = [10, "10", true]
julia> m[1]
```
This is an Any-typed array. It's not legal to modify an array with an object not matching the expected type.
```
julia> m = [1,2,3]
julia> m[1] = false # converts to 0
julia> m[1] = "false" # Error
julia> m = Array(<type>, <count>) #deprecated
julia> m = Array{type}(dims)
```
Indices start from __1__
```
julia> pop!(m) # ! indicated mutating method
julia> length(m) # all elements
julia> push!(m, 1)
```
```
julia> m = ones(2,3,4)
# 3 dimensional array of ones
julia> s = size(m)
# returns a tuple (2,3,4)
julia> s = ndims(m) # dimensions
julia> i = indices(m) # or indices(m, dim)
# returns tuple of valid indices (Base.OneTo(last)) element
```
##### Control flow
###### For
```
for <vname> in begin:end
    <operate on vname>
end
# begin:end is an generator, not a sequence
for <vname> in collection
  <operate on vname>
end
```
###### While
```
while <condition>
    < body >
    # break, continue allowed
end
```
##### Functions
```
function <name>(<args>)
  <body>
  [return [value]]
end
```
_nothing_ is eq to void (C), None(Python)
