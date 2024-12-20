/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///
#include "obstacle.h"

Rectangle::Rectangle(Buffers &_u, int row, int col, int _h, int _w):
    Obstacle(_u, row, col), height(_h), width(_w) {

    for (int r = row; r<row + height; r++){
	for (int c = col; c<col+width; c++){
	    if (u.chkBounds(r, c)){
		std::pair<int, int>lCoor = u.mapToLocal(r, c);
		*(u.alp(lCoor.first, lCoor.second)) = 0.0;
	    }
	}
    }
}

