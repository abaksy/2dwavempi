/// cse260 - hw 3
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// Provide grahics output
///
#ifndef __PLOTTING_H
#define __PLOTTING_H
#include "buffers.h"
#include "controlblock.h"
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <mpi.h>
#include <cstdio>


class Plotter {
public:
    Plotter(Buffers &_u, ControlBlock& _cb);
    ~Plotter();
    void updatePlot(int niter, int m, int n);

protected:

    Buffers &u;
    ControlBlock& cb;
    const int NDIMS;;  // time, row, col
  // netcdf stuff
    int ncid;
    int varid;
    size_t startp[3];
    size_t countp[3];
    int dimids[3];
    int tick;

    std::vector<double> globalPlot;




    /// for debug
    void printGlobal(ControlBlock&, int, int);
};

#endif
