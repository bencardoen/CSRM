from mpich.mpi4py import MPI


def sayHello(comm):
    print("Process {}".format(comm.Get_rank()))


def communicate(communicator):
    received = []
    if communicator.Get_rank() == 1:
        print("Sending ")
        communicator.send("abc", dest=0, tag=1)
    if communicator.Get_rank() == 0:
        print("Waiting for receipt")
        received = communicator.recv(source=1, tag=1)
        print("Received {}".format(received))

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    sayHello(comm)
    communicate(comm)

