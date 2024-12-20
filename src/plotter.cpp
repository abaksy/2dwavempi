/// cse260 - hw 3
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// provide grpahical output as a netcdf file
///
#include <iostream>
#include <vector>
#include "controlblock.h"
#include "plotter.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include <cstring>
#include <string>
#include <stdio.h>
#include <netcdf.h>
using namespace std;
#define ERRCODE 2
#define ERR(X)                                                                     \
    {                                                                              \
        fprintf(stderr, "Error: %s  %s %d\n", nc_strerror(X), __FILE__, __LINE__); \
        exit(ERRCODE);                                                             \
    }
Plotter::Plotter(Buffers &_u, ControlBlock &_cb) : u(_u),
                                                   cb(_cb), NDIMS(3), tick(0)
{
    if (cb.plot_freq == 0)
        return;

#ifdef _MPI_
        //
        // magic stuff goes here
        //
#endif

    for (int i = 0; i < cb.m * cb.n; i++)
    {
        globalPlot.push_back(0.0);
    }

    int retval, x_dimid, y_dimid, rec_dimid;
    string outName(cb.programPath.filename());
    outName = outName + ".nc";
    if ((retval = nc_create(outName.c_str(), NC_CLOBBER, &ncid)))
    {
        ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "y", cb.m, &y_dimid)))
    {
        ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "x", cb.n, &x_dimid)))
    {
        ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &rec_dimid)))
    {
        ERR(retval);
    }

    dimids[0] = rec_dimid;
    dimids[1] = y_dimid;
    dimids[2] = x_dimid;

    if ((retval = nc_def_var(ncid, "data", NC_DOUBLE, NDIMS, dimids, &varid)))
    {
        ERR(retval);
    }

    startp[0] = tick; // this is the time dimension.
    startp[1] = 0;    // this is the row dimension start
    startp[2] = 0;    // this is the col dimension start
    countp[0] = 1;
    countp[1] = cb.m; // # of elements in a col
    countp[2] = cb.n; // # of elements in a row

    if ((retval = nc_enddef(ncid)))
    {
        ERR(retval);
    }
}

///
/// update the plot output
///
void Plotter::updatePlot(int niter, int m, int n)
{

#ifndef _MPI_
    for (int i = 1, row = 0; i < u.M + 1; i++, row++)
    {
        for (int j = 1, col = 0; j < u.N + 1; j++, col++)
        {
            globalPlot[row * cb.n + (col)] = u.nxtV(i, j);
        }
    }
#else //_MPI_
    /// use MPI to gather data from processes
#endif

#ifdef _MPI_
    if (u.myRank == 0)
    {
#endif
        int retval;
        startp[0] = tick++;
        if ((retval = nc_put_vara_double(ncid, varid, startp, countp, &globalPlot.data()[0])))
        {
            ERR(retval);
        }
#ifdef _MPI_
    }
#endif
}

///
/// print the global buffer as ascii
///
void Plotter::printGlobal(ControlBlock &cb, int myRank, int iter)
{
    printf("%d  %5d--------------------------------\n", myRank, iter);
    double *p = &globalPlot.data()[0];
    for (int i = 0; i < cb.m; i++)
    {
        for (int j = 0; j < cb.n; j++)
        {
            printf("%2.3f ", *p++);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

Plotter::~Plotter()
{

    if (cb.plot_freq == 0)
        return;

    int retval;
    if (u.myRank == 0 && (retval = nc_close(ncid)))
    {
        ERR(retval);
    }
    if (u.myRank != 0)
    {
        return;
    }
}
