from mpich.mpi4py import MPI
import numpy
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# objects is what we will distribute
objects = ["a", "b", "c"] if rank == 0 else None

# Scatter and receive (each len(objects)/n)
print("Scattering {} for rank {}".format(objects, rank))
localobj = comm.scatter(objects, root=0)
print("rval for {} is {}".format(rank, localobj))

# Check if we need a barrier, gather is blocking, it will wait until all have called i
if rank == 1:
    time.sleep(2)
# Some trivial work
old, localobj = localobj, 'x'

print("modifying obj from {} to {}".format(old, localobj))

#print("@ barrier")
#comm.barrier()
#print("Passed barrier")

# Get the results back (all must call this)
objects = comm.gather(sendobj=localobj, root=0)
print("Received {}".format(objects))
