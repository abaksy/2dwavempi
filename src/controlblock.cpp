/// 
/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///
#include "controlblock.h"
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;
ControlBlock::ControlBlock(int argc, char *argv[]) :
    programPath(argv[0]),
    configFileName("wave260.config"),
    m(100), n(100),
    stats_freq(0),
    plot_freq(0),
    px(1), py(1),
    noComm(false),
    niters(100),
    gdbhack(false),
    wait(false),
    aofs(0) {

    /// options can be set from the config file
    /// and overridden by command line arguments
    /// Command line arguments
    /// Default value of the domain sizes
    static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
	{"niters", required_argument, 0, 'i'},
        {"stats-freq", required_argument, 0, 's'},
        {"plot", required_argument, 0, 'p'},
	{"px", required_argument, 0, 'x'},
	{"py", required_argument, 0, 'y'},
	{"aofs", no_argument, &aofs, 1},
	{"nocomm", no_argument, 0, 'k'},
	{"gdbhack", no_argument, 0, 'd'},

    };



    int nprocs=1, myRank=0;
#ifdef _MPI_
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
#endif

    if (argv[1][0] != '-'){
	// assume first argument is config file name
	configFileName = argv[1];
	if (myRank == 0)
	    cout << "Running config: " << configFileName << '\n';
    }

    ifstream f(configFileName);
    if (f.good()){
	config = json::parse(f);
	if (config.contains("-n")){
	    n = config["-n"];
	    m = n;
	}
	if (config.contains("-i")){
	    niters = config["-i"];
	}
	if (config.contains("-x")){
	    px = config["-x"];
	}
	if (config.contains("-y")){
	    py = config["-y"];
	}
    }
    for(int ac=1;ac<argc;ac++) {
	int c;
	while ((c=getopt_long(argc,argv,"n:i:s:x:y:p:PkdS:t",long_options,NULL)) != -1){
	    switch (c) {
	    case 0:
		break;

	    case 'n':
		// Size of the computational box
		n = atoi(optarg);
		m = n;
		break;

	    case 'x':
		//
		// X processor geometry
		px = atoi(optarg);
		break;

	    case 'y':
		// Y processor geometry
		py = atoi(optarg);
		break;
		
	    case 'i':
		// # of iterations
		// Use this option control the number of mesh sweeps
		niters = atoi(optarg);
		break;
		
	    case 's':
		// Print statistics for assessing correctness
		stats_freq = atoi(optarg);
		break;
		
		
	    case 'p':
		// Plot the excitation variable
		plot_freq = atoi(optarg);
		break;
		
	    case 'd':
		// Debug ouput
		gdbhack = true;
		break;
		
	    case 'k':
		// Shut off communication
		noComm = true;
		break;
	

		
		// Error
	    default:
		std::cout << "Usage: " << argv[0] << " [-n <domain size>] [-i <# iterations>]";
		std::cout << "\n\t    ";
		std::cout << " [-s <stats frequency>][-p <plot frequency>]\n\t";
		std::cout << "     [-x <x processor geometry>]";
		std::cout << "     [-y <x processor geometry>]";
		std::cout << "     [-d <gdbhack>]";
		std::cout << "     [-k <no communication>]" << endl;
		std::cout << "     [--aofs <use array of structs, default struct of arrays>]";
		std::cout << endl;
		exit(-1);
	    }
	}
    }


#ifdef _MPI_
    if ((px * py) != nprocs){
	if (!myRank){
	    if ((px * py) > nprocs)
		std::cout << "\n *** The number of processes in the geometry (" <<
		    px * py <<
		    ")\n     is larger than the number of available cores (" <<
		    nprocs << ")" << endl << endl;
	    else
		std::cout << "\n *** The number of processes in the geometry (" <<
		    px * py <<
		    ")\n     is smaller than the number of avilable cores (" <<
		    nprocs << ")" << endl << endl;
	}
	
	MPI_Finalize();
	exit(-1);
    }
#else
    if ((px * py) > 1){
	if (!myRank)
	    std::cout << "\n *** The number of processes in the geometry > 1, but you have not enabled MPI\n";
	exit(-1);
    }
#endif
}


    
