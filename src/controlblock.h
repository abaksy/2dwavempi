/// cse260
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
///

#pragma once
#include <filesystem>
#include <string>
#include "json.hpp"
using json = nlohmann::json;
using namespace std;
class ControlBlock {
 public:
    ControlBlock(int argc,  char *argv[]);
    std::filesystem::path programPath;
  
    string configFileName;
    json config;

    int m,n;
    int stats_freq;
    int plot_freq;
    int px, py;
    bool noComm;
    int niters;
    bool gdbhack;
    bool wait;
    int aofs;
};
