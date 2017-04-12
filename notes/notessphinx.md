# Sphinx notes
### Documenting a function
```
def <name>(arg1:int, arg2:list):
    """
    Oneliner explaining function
    .. math:
          <math here>
    
    :param [type] parameter: explain parameter
    :returns [type]: explain return value

    """
    return 0
```

Adding new modules (e.g. rst files)
```Bash
$sphinx-apidoc -o ./doc ./src
```
