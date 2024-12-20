/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// Provide stats output
///
#include "stats.h"
#include <stdio.h>
#include <math.h>
#include <chrono>


#ifdef _MPI_
#include <mpi.h>
#endif
#include "buffers.h"
Stats::Stats(Buffers & _u, ControlBlock &_cb, int _myRank, int _M, int _N):
  u(_u), cb(_cb), myRank(_myRank), M(_M), N(_N)
{
}

///
/// calcStats
///
/// calculate the local maximum and
/// the local sum of squares
///
void Stats::calcStats(){
    int debR = 0;
    int debC = 0;
    maxV = -1.0;
    sumSq = 0.0;
    for (int i=1; i<u.M+1; i++){
	for (int j=1; j<u.N+1; j++){
	    double uPos = fabs(u.curV(i, j));
	    if (uPos > maxV){
		maxV = uPos;
		debR = i;
		debC = j;
	    }
	    sumSq += uPos * (uPos);
	}
    }
    //    printf("DEBUG rank %d max(%d, %d) = %.12f\n", u.myRank, debR, debC, maxV);
    //    fflush(stdout);
}

///
/// print the summary statistics
///
void Stats::printStats(int iterations) {

    // elapsed is in milliseconds.
    calcStats();


    double globalMaxV = 0.0;;
    double globalSumSq = 0.0;

#ifndef _MPI_
    globalMaxV = maxV;
    globalSumSq = sumSq;
#endif

#ifdef _MPI_
    //
    //
    // collect the results and reduce into the
    // variables globalMaxV and globalSumSq
    //
    MPI_Reduce(&maxV, &globalMaxV, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&sumSq, &globalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#endif
    if (myRank == 0){
	auto rightNow = std::chrono::high_resolution_clock::now(); 
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(rightNow - startTime);
	//	double eTime = (double)elapsed/1000.0;
	double eTime = (double)elapsed.count()/1000.0;
	printf("@ %d x %d (%d = %d x %d) Max = %5.12f\tSumSq = %6.12f %d iterations in %4.3f seconds (Mpts/sec = %5.3f) %s\n",
	       M, N, cb.py*cb.px, cb.py, cb.px,
	       globalMaxV, globalSumSq/(double)(cb.m * cb.n), iterations, eTime, ((double)(cb.m/1000.0 * cb.n/1000.0 * iterations)/(double)eTime), cb.noComm ? "noComm" : "");

    }

}
