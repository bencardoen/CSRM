# MPI-Python notes

Notes taken from, amongst other, http://materials.jeremybejarano.com/MPIwithPython/

## Installation
You need a reference implementation of MPI, I use MPICH. Then you need python3 bindings, I pick mpi4py.
*here be dragons*
### Fedora 25
```
sudo dnf install mpich mpich-devel python3-mpi4py
```
Add the */bin/* path of the mpich install to your $PATH$ env variable
```
$rpm -ql mpich | grep mpieexec
```
Will provide the path, then add to ~/.bashrc
```
PATH=<yourmpichpath/bin/>:$PATH
```
This needs a reloading of your .bashrc file
```
$source ~/.bashrc
```
## Manual
This requires building MPICH and then installing/building mpi4py. Both are documented at their respective sites.

```
$ sudo pip3 install mpi4py # make sure MPICH paths are set
```

## Hello world in MPI-Python
```python3
# test.py
from mpich.mpi4py import MPI

comm=MPI.COMM_WORLD # static reference to communicator object
print("I'm process {}".format(comm.Get_rank()))
```
```bash
$mpiexec -n 3 python3 test.py
```

Get_size(), Get_rank() returns process count and current id.

## Communication a-b
```python3
obj = comm.recv(source:int=0, tag:int=0, status:Status=None) # Recv(buffer, source, tag, status) : buffer is byte object, recv : use pickle
comm.Send(buf|object, dest:int=0, tag:int=0) # send uses pickle, Send(buffer,...) takes a byte buffer object
# Blocking calls, wait until received. Use I{Receive|Send} for async.
# source=MPI.ANY_SOURCE == listening to all
```
Tag : if (data, src) eq for multiple messages, tag differentiates.

## Comm patterns
### [All]Reduce
```
comm.Reduce(sendbuffer, recvbuffer, op:Op=MPI.SUM, root:int=0) # allreduce implies root 0, allreduce if all processes need the answer, else only root has it
```

### BCast, Scatter
```
BCast(buf, root=0) # send buf as root to all
Scatter(sendbuf, recvbuf, root) # Same as bcast, but splits sendbuf into n recvbuffer parts (n processes), use Scatterv/Gatherv if data isn't evenly divided
```
```
from mpich.mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# objects is what we will distribute
objects = ["a", "b", "c"] if rank == 0 else None

# Scatter and receive (each len(objects)/n)
print("Scattering {} for rank {}".format(objects, rank))
localobj = comm.scatter(objects, root=0)
print("rval for {} is {}".format(rank, localobj))

# Some trivial work
old, localobj = localobj, 'x'

print("modifying obj from {} to {}".format(old, localobj))
print("@ barrier")
comm.barrier()
print("Passed barrier")

# Get the results back (all must call this)
objects = comm.gather(sendobj=localobj, root=0)
print("Received {}".format(objects))
```
