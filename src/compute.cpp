#include "stimulus.h"
#include "obstacle.h"
#include "compute.h"
#include "plotter.h"
#include <list>
#include <random>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

#ifdef _MPI_
#include <mpi.h>
#include <emmintrin.h>
#endif
#include <math.h>

#define CORNER_SIZE 1
#define PAD_SIZE 2

enum
{
    NORTH = 0,
    EAST,
    WEST,
    SOUTH
};

// Allocate space for sending and receiving ghost cells along east and west boundaries 
// (i.e. columns of the grid)
// Because the ghost cells are noncontiguous along columns, we need to pack
// and unpack the data when sending and receiving. 
// https://stackoverflow.com/questions/29121351/mpi-ghost-cell-exchange-with-mpi-type-create-subarray-or-mpi-type-vector-contigo/29134807#29134807
// Suggestion to not use MPI vector types as it doesn't offer much perf improvement
double *in_W, *in_E, *out_W, *out_E;

///
/// Compute object
///
/// upon construction, runs a simulation.
///
Compute::Compute(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
                 const int seedv) : u(u), plt(plt), cb(cb), myRank(_myRank), seedv(seedv), M(u.M), N(u.N), RANDSTIM(0)
{
    int my_pi = 0, my_pj = 0;
#ifdef _MPI_
    // my_pi, my_pj: position of the process in the x-by-y grid of processes (we distribute the processes in row major order)
    my_pi = myRank / cb.px;
    my_pj = myRank % cb.px;
    // Set top and bottom global edges based on my_pi
    if (my_pi == 0)
        topGlobalEdge = true;
    if (my_pi == cb.py - 1)
        botGlobalEdge = true;

    // Set left and right global edges based on my_pj
    if (my_pj == 0)
        leftGlobalEdge = true;
    if (my_pj == cb.px - 1)
        rightGlobalEdge = true;
#else
    // Uniprocessor case, the one processor is all 4 edges!
    topGlobalEdge = true;
    botGlobalEdge = true;
    leftGlobalEdge = true;
    rightGlobalEdge = true;
#endif
}

///
/// Simulate
///
/// calls class specific calcU and calcEdgeU
///
void Compute::Simulate()
{
    const unsigned int t = 1; // timestep
    const unsigned int h = 1; // grid space
    const double c = 0.29;    // velocity
    const double kappa = c * t / h;

    mt19937 generator(seedv);

    uniform_int_distribution<int> randRow(1, cb.m - 10);
    uniform_int_distribution<int> randCol(1, cb.n - 10);
    uniform_int_distribution<int> randEvent(0, 100);

    u.setAlpha((c * t / h) * (c * t / h));

    list<Stimulus *> sList;
    list<Obstacle *> oList;
    int iter = 0;

    ifstream f(cb.configFileName);
    if (cb.config.count("objects"))
    {
        for (int i = 0; i < cb.config["objects"].size(); i++)
        {
            auto const &ob = cb.config["objects"][i];
            if (ob["type"] == "sine")
            {
                sList.push_back(new StimSine(u, ob["row"], ob["col"],
                                             ob["start"], ob["duration"],
                                             ob["period"]));
            }
            else if (ob["type"] == "rectobstacle")
            {
                oList.push_back(new Rectangle(u, ob["row"], ob["col"],
                                              ob["height"], ob["width"]));
            }
        }
    }
    else
    {
        fprintf(stderr, "Using hardcoded stimulus\n");
        Rectangle obstacleA(u, cb.m / 2 + 5, cb.n / 2, 45, 5);
        Rectangle obstacleB(u, cb.m / 2 - 50, cb.n / 2, 45, 5);
        sList.push_back(new StimSine(u, cb.m / 2, cb.n / 3, 0 /*start*/, 500 /*duration*/, 10 /*period*/));
    }

    int messageCounter, i, j;

#ifdef _MPI_
    in_W  = new double[4*(u.M)];
    in_E  = in_W  + (u.M);
    out_W = in_E  + (u.M);
    out_E = out_W + (u.M);
#endif
    ///
    /// generate stimulus
    ///
    /// once quiet (non-deterministic),
    /// we exit this loop and go into a loop that
    /// continues until iterations is exhausted
    ///
    while (!sList.empty() && iter < cb.niters)
    {
        for (auto it = begin(sList); it != end(sList);)
        {
            if (!(*it)->doit(iter))
            {
                delete *it;
                it = sList.erase(it);
            }
            else
            {
                it++;
            }
        }
#ifdef _MPI_
        // Perform Ghost Cell exchanges
        messageCounter = 0;

        if (!cb.noComm)
        {
            // North Exchange
            if (!topGlobalEdge)  // Send the NORTH boundary & fill the NORTH ghost cells
            {
                MPI_Irecv(u.cur(0, CORNER_SIZE), u.N, MPI_DOUBLE, myRank - cb.px, SOUTH, MPI_COMM_WORLD, rcvRqst + messageCounter);
                MPI_Isend(u.cur(CORNER_SIZE, CORNER_SIZE), u.N, MPI_DOUBLE, myRank - cb.px, NORTH, MPI_COMM_WORLD, sndRqst + 0);

                messageCounter += 1;
            }

            // South Exchange
            if (!botGlobalEdge)  // Send the SOUTH boundary & fill the SOUTH ghost cells
            {
                MPI_Irecv(u.cur(u.M + CORNER_SIZE, CORNER_SIZE), u.N, MPI_DOUBLE, myRank + cb.px, NORTH, MPI_COMM_WORLD, rcvRqst + messageCounter);
                MPI_Isend(u.cur(u.M, CORNER_SIZE), u.N, MPI_DOUBLE, myRank + cb.px, SOUTH, MPI_COMM_WORLD, sndRqst + 1);

                messageCounter += 1;
            }

            // West Exchange
            if (!leftGlobalEdge) // Send the WEST boundary & fill the WEST ghost cells 
            {
                for(i = 1; i <= u.M; ++i)
                {
                    out_W[i-1] = u.curV(i, 1);   
                }

                MPI_Irecv(in_W, u.M, MPI_DOUBLE, myRank - 1, EAST, MPI_COMM_WORLD, rcvRqst + messageCounter);
                MPI_Isend(out_W, u.M, MPI_DOUBLE, myRank - 1, WEST, MPI_COMM_WORLD, sndRqst + 2);
            
                messageCounter += 1;
            }

            // East Exchange
            if (!rightGlobalEdge)
            {
                for(i = 1; i <= u.M; ++i)
                {
                    out_E[i-1] = u.curV(i, u.gridN-2);   
                }

                MPI_Irecv(in_E, u.M, MPI_DOUBLE, myRank + 1, WEST, MPI_COMM_WORLD, rcvRqst + messageCounter);
                MPI_Isend(out_E, u.M, MPI_DOUBLE, myRank + 1, EAST, MPI_COMM_WORLD, sndRqst + 3);

                messageCounter += 1;
            }
        }
#endif

        calcU(u);

#ifdef _MPI_
    if(!cb.noComm)
    {   
        MPI_Waitall(messageCounter, rcvRqst, recvStatus.begin());

        if (!leftGlobalEdge)
        {
            // Populate values from in_W
            for(i = 1; i < u.gridM-1; ++i)
            {
                *(u.cur(i, 0)) = in_W[i-1];
            }
            
        }
        

        if (!rightGlobalEdge)
        {
            // Populate values from in_E
            for(i = 1; i < u.gridM-1; ++i)
            {
                *(u.cur(i, u.gridN-1)) = in_E[i-1];
            }
            
        }
        
    }
#endif

        calcEdgeU(u, kappa);

        if (cb.plot_freq && iter % cb.plot_freq == 0)
            plt->updatePlot(iter, u.gridM, u.gridN);

        // DEBUG start
        //	u.printActive(iter);
        // DEBUG end

        u.AdvBuffers();

        iter++;
    }

    ///
    /// all stimulus done
    /// keep simulating till end
    ///
    for (; iter < cb.niters; iter++)
    {
#ifdef _MPI_
        // Perform Ghost Cell exchanges
        messageCounter = 0;
        MPI_Request send_req2[8];
        MPI_Request rcv_req2[8];
        MPI_Status msgStatus2[4];

        if (!cb.noComm)
        {
            // North Exchange
            if (!topGlobalEdge)  // Send the NORTH boundary & fill the NORTH ghost cells
            {
                MPI_Irecv(u.cur(0, CORNER_SIZE), u.N, MPI_DOUBLE, myRank - cb.px, SOUTH, MPI_COMM_WORLD, rcv_req2 + messageCounter);
                MPI_Isend(u.cur(CORNER_SIZE, CORNER_SIZE), u.N, MPI_DOUBLE, myRank - cb.px, NORTH, MPI_COMM_WORLD, send_req2);

                messageCounter += 1;
            }

            // South Exchange
            if (!botGlobalEdge)  // Send the SOUTH boundary & fill the SOUTH ghost cells
            {
                MPI_Irecv(u.cur(u.M + CORNER_SIZE, CORNER_SIZE), u.N, MPI_DOUBLE, myRank + cb.px, NORTH, MPI_COMM_WORLD, rcv_req2 + messageCounter);
                MPI_Isend(u.cur(u.M, CORNER_SIZE), u.N, MPI_DOUBLE, myRank + cb.px, SOUTH, MPI_COMM_WORLD, send_req2 + 1);

                messageCounter += 1;
            }
            

            // West Exchange
            if (!leftGlobalEdge)
            {
                for(i = 1; i < u.gridM - 1; ++i)
                {
                    out_W[i-1] = u.curV(i, 1);   
                }

                MPI_Irecv(in_W, u.M, MPI_DOUBLE, myRank - 1, EAST, MPI_COMM_WORLD, rcv_req2 + messageCounter);
                MPI_Isend(out_W, u.M, MPI_DOUBLE, myRank - 1, WEST, MPI_COMM_WORLD, send_req2 + 2);
            
                messageCounter += 1;
            }

            

            // East Exchange
            if (!rightGlobalEdge)
            {
                for(i = 1; i < u.gridM - 1; ++i)
                {
                    out_E[i-1] = u.curV(i, u.gridN-2);   
                }

                MPI_Irecv(in_E, u.M, MPI_DOUBLE, myRank + 1, WEST, MPI_COMM_WORLD, rcv_req2 + messageCounter);
                MPI_Isend(out_E, u.M, MPI_DOUBLE, myRank + 1, EAST, MPI_COMM_WORLD, send_req2 + 3);

                messageCounter += 1;
            }
            
        }
#endif
        calcU(u);
            if (cb.plot_freq && iter % cb.plot_freq == 0)
                plt->updatePlot(iter, u.gridM, u.gridN);

#ifdef _MPI_
    if(!cb.noComm)
    {   
        MPI_Waitall(messageCounter, rcv_req2, msgStatus2);
        
        if (!leftGlobalEdge)
        {
            // Populate values from in_W
            for(i = 1; i < u.gridM-1; ++i)
            {
                *(u.cur(i, 0)) = in_W[i-1];
            }
        }
        

        if (!rightGlobalEdge)
        {
            // Populate values from in_E
            for(i = 1; i < u.gridM-1; ++i)
            {
                *(u.cur(i, u.gridN-1)) = in_E[i-1];
            }
        }
    }
#endif

        calcEdgeU(u, kappa);
        if ((cb.plot_freq != 0) && (iter % cb.plot_freq == 0))
            plt->updatePlot(iter, u.gridM, u.gridN);

        // DEBUG
        // u.printActive(iter);
        u.AdvBuffers();
    }
#ifdef _MPI_
    delete[] in_W;
#endif
}

TwoDWave::TwoDWave(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
                   const int seedv) : Compute(u, plt, cb, _myRank, seedv) {};

///
/// compute the interior cells
///
///
#ifndef _MPI_
void TwoDWave::calcU(Buffers &u)
{
    // interior always starts at 2,2, ends at gridN
    for (int i = 2; i < u.gridM - 2; i++)
    {
        for (int j = 2; j < u.gridN - 2; j++)
        {
            *u.nxt(i, j) =
                u.alpV(i, j) *
                    (u.curV(i - 1, j) + u.curV(i + 1, j) +
                     u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
                2 * u.curV(i, j) - u.preV(i, j);
        }
    }
}
#else 
void TwoDWave::calcU(Buffers &u)
{
    __m128d cc, nn, ee, ww, ss, UU; // vectorized stencil
    __m128d pp, pp2, cc2; // values of previous iteration
    __m128d alpha_alpha;
    __m128d four_four = _mm_set1_pd(4.0);
    __m128d two_two = _mm_set1_pd(2.0);

    // interior always starts at 2,2, ends at gridN
    for (int i = 2; i < u.gridM - 2; i++)
    {
        for (int j = 2; j < u.gridN - 2; j+=2)
        {
            alpha_alpha = _mm_loadu_pd(u.alp(i, j));
            ww = _mm_loadu_pd(u.cur(i-1, j)); // West neighbour
            cc = _mm_loadu_pd(u.cur(i, j));
            ee = _mm_loadu_pd(u.cur(i+1, j));
            nn = _mm_loadu_pd(u.cur(i, j-1));
            ss = _mm_loadu_pd(u.cur(i, j+1));
            pp = _mm_loadu_pd(u.pre(i, j));
            cc2 = _mm_mul_pd(two_two, cc);
            UU = _mm_sub_pd(_mm_add_pd(cc2, _mm_mul_pd(alpha_alpha, _mm_sub_pd(
                    _mm_add_pd(ww, 
                        _mm_add_pd(ee,
                         _mm_add_pd(nn, ss))), 
                         _mm_mul_pd(four_four, cc)))), pp);
            _mm_storeu_pd(u.nxt(i, j), UU);

            // *u.nxt(i, j) =
            //     u.alpV(i, j) *
            //         (u.curV(i - 1, j) + u.curV(i + 1, j) +
            //          u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
            //     2 * u.curV(i, j) - u.preV(i, j);
        }
    }
}

#endif

///
/// compute edges
///
/// compute interior edges. These are not ghost cells but cells that rely
/// on either ghost cell values or boundary cell values.
///
void TwoDWave::calcEdgeU(Buffers &u, const double kappa)
{

#ifdef _MPI_
    __m128d cc, nn, ee, ww, ss, UU; // vectorized stencil
    __m128d pp, pp2, cc2; // values of previous iteration
    __m128d alpha_alpha;
    __m128d four_four = _mm_set1_pd(4.0);
    __m128d two_two = _mm_set1_pd(2.0);
    int stride = 2;
#else 
    int stride = 1;
#endif

    // top and bottom edge
    for (int j = 1; j < u.gridN - 1; j+=stride)
    {
        int i = 1;
#ifdef _MPI_
        alpha_alpha = _mm_loadu_pd(u.alp(i, j));
        ww = _mm_loadu_pd(u.cur(i-1, j)); // West neighbour
        cc = _mm_loadu_pd(u.cur(i, j));
        ee = _mm_loadu_pd(u.cur(i+1, j));
        nn = _mm_loadu_pd(u.cur(i, j-1));
        ss = _mm_loadu_pd(u.cur(i, j+1));
        pp = _mm_loadu_pd(u.pre(i, j));
        cc2 = _mm_mul_pd(two_two, cc);
        UU = _mm_sub_pd(_mm_add_pd(cc2, _mm_mul_pd(alpha_alpha, _mm_sub_pd(
                _mm_add_pd(ww, 
                    _mm_add_pd(ee,
                     _mm_add_pd(nn, ss))), 
                     _mm_mul_pd(four_four, cc)))), pp);
        _mm_storeu_pd(u.nxt(i, j), UU);
#else
        *u.nxt(i, j) =
            u.alpV(i, j) *
                (u.curV(i - 1, j) + u.curV(i + 1, j) +
                 u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
#endif
        i = u.gridM - 2;

#ifdef _MPI_
        alpha_alpha = _mm_loadu_pd(u.alp(i, j));
        ww = _mm_loadu_pd(u.cur(i-1, j)); // West neighbour
        cc = _mm_loadu_pd(u.cur(i, j));
        ee = _mm_loadu_pd(u.cur(i+1, j));
        nn = _mm_loadu_pd(u.cur(i, j-1));
        ss = _mm_loadu_pd(u.cur(i, j+1));
        pp = _mm_loadu_pd(u.pre(i, j));
        cc2 = _mm_mul_pd(two_two, cc);
        UU = _mm_sub_pd(_mm_add_pd(cc2, _mm_mul_pd(alpha_alpha, _mm_sub_pd(
                _mm_add_pd(ww, 
                    _mm_add_pd(ee,
                     _mm_add_pd(nn, ss))), 
                     _mm_mul_pd(four_four, cc)))), pp);
        _mm_storeu_pd(u.nxt(i, j), UU);
#else 
        *u.nxt(i, j) =
            u.alpV(i, j) *
                (u.curV(i - 1, j) + u.curV(i + 1, j) +
                 u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
#endif
    }

    // left and right
    for (int i = 1; i < u.gridM - 1; i++)
    {
        int j = 1;
        *u.nxt(i, j) =
            u.alpV(i, j) *
                (u.curV(i - 1, j) + u.curV(i + 1, j) +
                 u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
        j = u.gridN - 2;
        *u.nxt(i, j) =
            u.alpV(i, j) *
                (u.curV(i - 1, j) + u.curV(i + 1, j) +
                 u.curV(i, j - 1) + u.curV(i, j + 1) - 4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
    }

    // set the boundary conditions to absorbing boundary conditions (ABC)
    // du/dx = -1/c du/dt   x=0
    // du/dx = 1/c du/dt    x=N-1
    // conditions for an internal boundary (ie.g. ghost cells)
    // top edge

    // top global edge (instead of ghost cells)
    if (topGlobalEdge)
    {
        // top row absorbing boundary condition
        int i = 0;
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(i, j) = u.curV(i + 1, j) +
                           ((kappa - 1) / (kappa + 1)) * (u.nxtV(i + 1, j) - u.curV(i, j));
        }
    }

    // bottom edge (instead of ghost cells)
    if (botGlobalEdge)
    {
        int i = u.gridM - 1;
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(i, j) = u.curV(i - 1, j) +
                           ((kappa - 1) / (kappa + 1)) * (u.nxtV(i - 1, j) - u.curV(i, j));
        }
    }

    // left edge
    if (leftGlobalEdge)
    {
        int j = 0;
        for (int i = 1; i < u.gridM - 1; i++)
        {
            *u.nxt(i, j) = u.curV(i, j + 1) +
                           ((kappa - 1) / (kappa + 1)) * (u.nxtV(i, j + 1) - u.curV(i, j));
        }
    }
    // right edge
    if (rightGlobalEdge)
    {
        int j = u.gridN - 1;
        for (int i = 1; i < u.gridM - 1; i++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1) +
                           ((kappa - 1) / (kappa + 1)) * (u.nxtV(i, j - 1) - u.curV(i, j));
        }
    }
}

//!
//! Use a different propgation model
//! This model shifts values in the horizontal direction
//!
DebugPropagate::DebugPropagate(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
                               const int seedv) : Compute(u, plt, cb, _myRank, seedv) {};

//!
//! compute the interior cells
//!
void DebugPropagate::calcU(Buffers &u)
{

    // interior always starts at 2,2, ends at gridN-3
    for (int i = 2; i < u.gridM - 2; i++)
    {
        for (int j = 2; j < u.gridN - 2; j++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1);
        }
    }
}

//!
//! compute edges
//! (either interior edges or global edges)
//!
void DebugPropagate::calcEdgeU(Buffers &u, const double kappa)
{
    if (topGlobalEdge)
    {
        // top row absorbing boundary condition
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(1, j) = 0;
        }
    }
    else
    {
        int i = 1;
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1);
        }
    }

    // bottom edge
    if (botGlobalEdge)
    {
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(u.gridM - 2, j) = 0;
        }
    }
    else
    {
        int i = u.gridM - 2;
        for (int j = 1; j < u.gridN - 1; j++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1);
        }
    }

    // left edge
    if (leftGlobalEdge)
    {
        for (int i = 1; i < u.gridM - 1; i++)
        {
            *u.nxt(i, 1) = 0.0;
        }
    }
    else
    {
        int j = 1;
        for (int i = 1; i < u.gridM - 1; i++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1);
        }
    }
    // right edge
    if (rightGlobalEdge)
    {
        for (int i = 1; i < u.gridM - 1; i++)
        {
            // right column
            *u.nxt(i, u.gridN - 2) = 0.0;
        }
    }
    else
    {
        int j = u.gridN - 2;
        for (int i = 1; i < u.gridM - 1; i++)
        {
            *u.nxt(i, j) = u.curV(i, j - 1);
        }
    }
}
