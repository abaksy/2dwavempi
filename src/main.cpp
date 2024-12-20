#include "buffers.h"
#include "compute.h"
#include "plotter.h"
#include "controlblock.h"
#include "stats.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include <chrono>
#include <unistd.h>
///
/// cse260 - hw3
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///

int main(int argc, char *argv[]) {

#ifdef _MPI_
    MPI_Init(&argc, &argv);
#endif

    ControlBlock cb(argc, argv);


    //
    // you can use this hack to force your program to wait
    // use gdb -p <process #> to attach to a running program
    // then gdb> set var wt = 1
    //      gdb> cont 
    // see : https://www.open-mpi.org/faq/?category=debugging  
    //
    volatile int wt = 0;
    if (cb.gdbhack){
      printf("PID %d ready for attach\n", getpid());
      fflush(stdout);
      while (wt == 0){
	sleep(5);
      }
    }

    int myRank = 0;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
#endif

    Buffers *uValP;

    //
    // choose the basic datatype for
    // buffers.  Array of Structs or
    // Struct of Arrays
    // 
    if (cb.aofs){
	uValP = new AofSBuff(cb, myRank);
	if (myRank == 0)
	  printf("Using Array of Structs\n");
    }else{
	uValP = new ArrBuff(cb, myRank);
	if (myRank == 0)
	  printf("Using Struct of Arrays\n");
    }

    Buffers& uVal = *uValP;

    Plotter *plt = new Plotter(uVal, cb); 


    // DebugPropagate compute(uVal, plt, cb, myRank);
    TwoDWave compute(uVal, plt, cb, myRank);

    Stats stats(uVal, cb, myRank, cb.m, cb.n);

    stats.setStartTime();
    compute.Simulate();
    stats.printStats(cb.niters);

    // cleanup
    delete plt;
    delete uValP;
#ifdef _MPI_
    MPI_Finalize();
#endif


}
