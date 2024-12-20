/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// Provide stats output
///
#pragma once
#include <chrono>
#include "buffers.h"
#include "controlblock.h"
#include <chrono>

class Stats {
 public:
    Stats(Buffers &_u, ControlBlock &_cb, int myRank, int M, int N);
    void calcStats();
    void printStats(int);
    double getMaxV() {return maxV;};
    double getSumSq() {return sumSq;};
    void setStartTime() { if (myRank == 0) startTime = std::chrono::high_resolution_clock::now(); }
 protected:
    Buffers &u;
    ControlBlock &cb;
    int M;
    int N;
    int myRank;
    double maxV;
    double sumSq;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};
