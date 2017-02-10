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
##### Type system
Conversion isn't as loose as in C[++]. E.g. bool->int is legal, the reverse is not.

Parametric typing : use {T}
e.g.
```
m = Array{Int64,1} # 1 dimensional array of 64 bit signed int
typeof(Array(Int64, 1)) == Vector{Int64} # true
```
Note expression{T} is a Type, not a value.
```
Array(Int64, 1) == Vector{Int64} # false, lhs is not a type but an object
```

Numeric types:
Bool, Int{x}, Float{x} where x is [8,16,32,64], Complex{basetype}

Numeric operations
```
julia> x = 5
julia> y = 7
julia> (3x + 5y) / y^2
julia> +(x,y)
julia> (x+y)y # legal
julia> (y)(x+y) # Error (parsing precedence)
```
*type{min|max}* provides range of types.

*epsilon(...)* provides mach eps

*{next|prev}float(x)*

```
julia> 1.1 + 0.1 # 1.200002
julia> with_rounding(Type, Mode) do <code> end # Default is Float64, RoundNearest
```
*Big{Int|Float}*
```
julia> b = BigInt(value)
julia> b = parse(BigInt,"123")
```
*{zero|one}(x)* Returns 1,0 of any x type.

*===* Tests if @lhs == @rhs

Strings
```
julia> s = "42, surely? Or what do you think?"
# split(s[, delimiter] )
julia> s.replace(s, <orig>, <new> )
julia> s*s # concat
julia> strip(s) # remove whitespace
julia> match(<pattern>, <string> )
# returns matchobject
# pattern is of the form r"<regex>"
```

##### Tuples
Similar to Python
```
julia> m = "a", "b" # equiv to m=("a", "b")
julia> f, s = m # unpacking is supported
```

##### Arrays
```
julia> m = [10, "10", true]
julia> m[1]
```
This is an Any-typed array. It's not legal to modify an array with an object not matching the expected type.
```
julia> m = [1,2,3] # flat array (contrast with [1 2 3] which is a 1x3)
julia> m[1] = false # converts to 0
julia> m[1] = "false" # Error
julia> m = Array(<type>, <count>) #deprecated
julia> m = Array{type}(dims)
julia> m = fill(value, dims,)
julia> m = [1 2; 3 4] # 2x2 array
julia> m = [1;2;3;4] # NOT an 4x1 vector, flat array (4 element array)
julia> m = [1 2 3 4]' # Transpose of 1x4 = 4x1 array
```
Indices start from __1__
```
julia> m = [1 2; 3 4]
julia> m[1] == 1
julia> m[1, 1] == 1
julia> m[1, :] == [1; 2]
julia> b = [true false; false true]
julia> m[b] == [1;4]
julia> m[2:3] = 42 # [1 42; 42; 4]
```
Operations
```
julia> pop!(m) # ! indicated mutating method
julia> length(m) # all elements
julia> push!(m, 1)
julia> sort(m, rev=true) # reverse sorted copy of m
# Others include {max|min}imum , mean, std, var
```
Multidimensional (Vector = 1 dim, Matrix is n)
```
julia> m = ones(2,3,4)
# 3 dimensional array of ones
julia> s = size(m)
# returns a tuple (2,3,4)
julia> s = ndims(m) # dimensions
julia> i = indices(m) # or indices(m, dim)
# returns tuple of valid indices (Base.OneTo(last)) element
julia> m[i:j] # slicing (returns (i,j) inclusive)
julia> m = [1,2,3,4]
julia> b = reshape(m,2,2) # b is a 2x2 representation of a, if you modify either you modify both
julia> b = b[:] # slice copy, destroys view gives original array
julia> b = squeeze(b, dim)[:] #
```
Matrix Operations
```
julia> c = [1 0 ; 0 1]
julia> e = c' # transpose
julia> i = inv(c)
julia> b = [1 2 ; 3 4]
julia> a = [1 1 ; 1 1]
julia> b * a
#a x = b, solve for x
julia> a \ b
julia> dot(ones(3), ones(3))
julia> ones(2) * a # dim error (2x1 x 2x2)
julia> a * ones(2) # ok
julia> a .* 5 # element wise mult (.x where x is operator is element wise as in Matlab)
julia> 5a == a.*5 # scalar mult is unambiguous
```

_end_ keyword in a sequence reference last element

###### Dicts
```
julia> a = Dict(key => value [, key => value])
julia> a[key] # gets value
for d in a
    println(d.second) # value (first is key)
end
```
Mutating is done by accessor
```
for d in keys(a)
    a[d] = <newvalue>
end
```
Iterating over a and modifying pair in place will lead to an error.

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

Shorthand Functions
```
julia> f(x) = x^2
julia> x-> x^2
```
Use unnamed functions (l's)
```
julia> map(x->x^2, collection)
```
Arguments: same rules as in Python.
```
function f(first, second; third=named, fourth=named )
```

##### IO
```
julia> f = open(<fname>, mode) # mode is w,r
julia> write(f, <content>)
julia> readall(f)
julia> for q in eachline(f) println(q) end
```

##### Iterators
```
julia> a = Dict(1=>2, 2=>1)
julia> q = collect(keys(a))
```
Collect transforms iterator to collection.
```
julia> a= [1,2,3]
julia> b = [3,4,5]
julia> c, d, e = zip(a,b)
```
List comprehension (pythonic)
```
julia> a = [1,2,3]
julia> b = [i for i in a if i % 2 == 0]
```

##### Logic
Logical and/or/not  (bool only) : &&, ||, !
Bitwise : &, |, ~ (watch out for signedness UInt8 != Int8)
