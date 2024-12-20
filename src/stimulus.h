/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///
#include "buffers.h"
#include <math.h>
///
/// Stimulus base class
///
class Stimulus {
 public:
    Stimulus(Buffers &_u, int row, int col, int startTime, int duration):
	row(row), col(col), u(_u), startTime(startTime), duration(duration)
    {}
  virtual bool doit(int iter) = 0;
  virtual ~Stimulus() {};
 protected:
    Buffers &u;
    int startTime;  /// event starts at this time
    int duration;   /// event lasts this long
    int tick;
    int row;        /// event is at this row (global coord)
    int col;        /// event is at this col (global coord)
};

///
/// Sinusoidal stimulus
///
class StimSine: public Stimulus {
 public:
    StimSine(Buffers &_u, int row, int col, int startTime, int duration, int _period=8):
	Stimulus(_u, row, col, startTime, duration), period(_period) {};
    ~StimSine() {};
    bool doit(int);
 protected:
    int period;
    const int amplitude = 10;
    
};

///
/// Square wave stimulus
///
class StimSquare: public Stimulus {
 public:
    StimSquare(Buffers &_u, int row, int col, int startTime, int duration, int _period=1):
	Stimulus(_u, row, col, startTime, duration), period(_period) {}
    ~StimSquare() {};
  bool doit(int);
 protected:
    int period;
    const int amplitude = 2;
    
};

///
/// Pyramid collision
///
class StimPyramid: public Stimulus {
 public:
    StimPyramid(Buffers &_u, int row, int col, int startTime, int duration):
	Stimulus(_u, row, col, startTime, duration), r(0.0) {}
    ~StimPyramid() {};
  bool doit(int);
 protected:
    const int radius = 2;
    double r;
};
    
