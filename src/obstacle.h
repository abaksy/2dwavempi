/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///
#include "buffers.h"
#include <math.h>
///
/// Obstacle base class
///
class Obstacle {
 public:
    Obstacle(Buffers &_u, int row, int col):
	row(row), col(col), u(_u)
    {}
  virtual ~Obstacle() {};
 protected:
    Buffers &u;
    int row;        /// event is at this row (global coord)
    int col;        /// event is at this col (global coord)
};

///
/// Rectangle
///
class Rectangle: public Obstacle {
 public:
    Rectangle(Buffers &_u, int row, int col, int height, int width);
    ~Rectangle() {};
 protected:
    int height;
    int width;
};
