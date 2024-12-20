/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// basic buffers for the simulation region
///
#pragma once
#include "controlblock.h"
#include <vector>
using namespace std;


class Buffers{
 public:
    Buffers(ControlBlock& _cb, int _myRank);
    virtual ~Buffers() = default;
    virtual void AdvBuffers()  = 0;
    virtual void setAlpha(double aVal);
    void print(int);
    void printAlpha();
    void printMap(int);
    void printActive(int);
    double sumSq(int, int, int, int);


    // map global coordinate to local
    std::pair<int,int> mapToLocal(int r, int c);
    // map local coordinate to global
    std::pair<int,int> mapToGlobal(int r, int c);
    // check for in bounds (r and c are global coordinates NOT including ghost cells)
    bool chkBounds(int r, int c);


    // these accessors use local coordinates including ghost cells
    virtual double *cur(int, int) = 0; 
    virtual double curV(int, int) = 0; 
    virtual double *nxt(int, int) = 0; 
    virtual double nxtV(int, int) = 0; 
    virtual double *pre(int, int) = 0; 
    virtual double preV(int, int) = 0; 
    virtual double *alp(int r, int c) = 0;
    virtual double alpV(int r, int c) = 0;

    //
    // the remainder is [0, z-1] where z is the # of processors in the
    // dimension.
    //
    
    bool getExtraRow(){return getExtraRow(myRank/cb.px, cb.m, cb.py);}
    bool getExtraRow(int myRow, int m, int py){
	return((m % py) > (myRow));
    }
    bool getExtraCol(){return getExtraCol(myRank, cb.n, cb.px);}
    bool getExtraCol(int myRank, int n, int px){
	int myCol = myRank % px;
	return((n % px) > myCol);
    }


    std::vector<int> startRows;   // global start row coordinates (excluding ghost cells)
    std::vector<int> startCols;  // global start column coordinates (excluding ghost cells)

    int gridM;    // grid including ghost cells
    int gridN;    // grid including ghost cells
    int M;        // grid excluding ghost cells
    int N;        // grid excluding ghost cells
    int myRank;
    int my_pi, my_pj;

 protected:
    ControlBlock& cb;
    double *memoryPool;
};


///
/// separate arrays for each major
/// buffer
///
class ArrBuff : public Buffers
{
 public:
    ArrBuff(ControlBlock& _cb, int _myRank);
    ~ArrBuff();
    void AdvBuffers();

    inline double *cur(int r, int c){
	return(curr + r*gridN + c);
    }
    inline double curV(int r, int c){
	return *(curr + r*gridN + c);
    }
    inline double *nxt(int r, int c){
	return(next + r*gridN + c);
    }
    inline double nxtV(int r, int c){
	return *(next +r*gridN + c);
    }
    inline double *pre(int r, int c){
	return (prev + r*gridN + c);
    }
    inline double preV(int r, int c){
	return *(prev + r*gridN + c);
    }
    inline double *alp(int r, int c){
	return (alpha + r*gridN + c);
    }
    inline double alpV(int r, int c){
	return *(alpha + r*gridN + c);
    }

 protected:
    double *memoryPool;
    double *u0, *u1, *u2;    // sub buffers
    double *alpha;
    double *next;
    double *curr;
    double *prev;
};


///
/// one buffer, each entry is a struct
///
class AofSBuff : public Buffers
{
 public:
    AofSBuff(ControlBlock& _cb, int _myRank);
    ~AofSBuff();
    void AdvBuffers();

    /// make a point structure
    struct point {
	double x[4];
    };

    void print(int);
    inline double *cur(int r, int c){
	return(&memoryPool[r*gridN + c].x[current]);
    }
    inline double curV(int r, int c){
	return(memoryPool[r*gridN + c].x[current]);
    }
    inline double *nxt(int r, int c){
	return(&memoryPool[r*gridN + c].x[next]);
    }
    inline double nxtV(int r, int c){
	return(memoryPool[r*gridN + c].x[next]);
    }
    inline double *pre(int r, int c){
	return(&memoryPool[r*gridN + c].x[previous]);
    }
    inline double preV(int r, int c){
	return(memoryPool[r*gridN + c].x[previous]);
    }
    inline double *alp(int r, int c){
	return(&memoryPool[r*gridN + c].x[alpha]);
    }
    inline double alpV(int r, int c){
	return(memoryPool[r*gridN + c].x[alpha]);
    }

 protected:
    struct point *memoryPool;
    int next;
    int current;
    int previous;
    const int alpha;
};
