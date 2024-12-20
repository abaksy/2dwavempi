/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///
#include "stimulus.h"
#include <stdio.h>
///
/// sine wave 
///
bool StimSine::doit(int iter){
    if (iter > startTime+duration){
	return false;
    }
    if (iter < startTime){
	return true;
    }
    if (iter == startTime){
	tick = 0;
    }
    if (u.chkBounds(row, col)){
	double v = amplitude * sin(2*M_PI * (double)tick/(double)period);
	// the row and col here are global #'s that don't include ghost cells
	// all arrays include ghost cells and use ghost cell coordinates.
	// so we need to add 1 to row and col.
	std::pair<int, int>lCoor = u.mapToLocal(row, col);
	*u.cur(lCoor.first, lCoor.second) = v;
	*u.pre(lCoor.first, lCoor.second) = v;
    }
    tick++;
    return true;
}

///
/// square wave
///
bool StimSquare::doit(int iter){
    if (iter > startTime+duration){
	return false;
    }
    if (iter < startTime){
	return true;
    }
    if (iter == startTime){
	tick = 0;
    }
    double v = 0.0;
    if (tick < period/2){
	v = (double)amplitude;
    }else if (tick < period){
	v = 0.0;
    }else{
	tick = 0;
	v = 0.0;
    }

    if (u.chkBounds(row, col)){
	// the row and col here are global #'s that don't include ghost cells
	// all arrays include ghost cells and use ghost cell coordinates.
	// so we need to add 1 to row and col.
	std::pair<int, int>lCoor = u.mapToLocal(row, col);
	//	printf("DEBUG: Stim at global w/o ghost( %d, %d)\n", row, col);
	//	printf("DEBUG: setting nxt and cur at %d %d to %.12f\n", lCoor.first+1, lCoor.second+1, v); 
	//	fflush(stdout);
	*u.nxt(lCoor.first+1, lCoor.second+1) = v;
	*u.cur(lCoor.first+1, lCoor.second+1) = v;
    }
    tick++;
    return true;
}


///
/// pyramid collision
///
bool StimPyramid::doit(int iter){
    if (iter > startTime+duration){
	return false;
    }
    if (iter < startTime){
	return true;
    }
    if (iter == startTime){
	tick = 0;
    }


    double rStep = ((double)radius/(double)duration);
    r += rStep;
    int h = (int)ceil(r);

    if (tick < duration){
	/// deform region of a solid square that is 2r x 2r
	/// in area
	///
	for (int rOff = -h; rOff <= h; rOff++){
	    for (int cOff = -h; cOff <= h; cOff++){
		//		*u.cur(row+rOff, col+cOff) -= rStep;
		if (u.chkBounds(row+rOff, col+cOff))
		    *u.cur(row+rOff, col+cOff) -= rStep;
	    }
	}
    }else{
	for (int rOff = -h; rOff <= h; rOff++){
	    for (int cOff = -h; cOff <= h; cOff++){
		if (u.chkBounds(row+rOff, col+cOff))
		    *u.cur(row+rOff, col+cOff) = 0.0;
	    }
	}
    }
    tick++;
    return true;
}

