cmake_minimum_required(VERSION 3.10...3.19)

project(wave260)
set(SOURCES 
	src/main.cpp
	src/plotter.cpp
	src/buffers.cpp
	src/compute.cpp
	src/stats.cpp
	src/controlblock.cpp
	src/stimulus.cpp
	src/obstacle.cpp
	)


add_executable(wave260mpi ${SOURCES})
add_executable(wave260uni ${SOURCES})

find_package(MPI REQUIRED)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
   find_package(netcdf REQUIRED)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
   set(netCDF_INCLUDE_PATH /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen2/gcc-10.2.0/netcdf-c-4.8.1-yt65vteujsipoyljuvot454iwmr6wwtd/include )
   set(netCDF_LIB_PATH     /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen2/gcc-10.2.0/netcdf-c-4.8.1-yt65vteujsipoyljuvot454iwmr6wwtd/lib )
   target_link_directories(wave260mpi PRIVATE ${netCDF_LIB_PATH})
   target_link_directories(wave260uni PRIVATE ${netCDF_LIB_PATH})
   set(netCDF_LIBRARIES -lnetcdf)
endif()


target_link_libraries(wave260mpi PUBLIC ${MPI_LIBRARIES} ${netCDF_LIBRARIES})
target_include_directories(wave260mpi PUBLIC ${MPI_INCLUDE_PATH} ${netCDF_INCLUDE_PATH})
target_compile_definitions(wave260mpi PRIVATE "_MPI_=1")


TARGET_link_libraries(wave260uni PUBLIC ${MPI_LIBRARIES} ${netCDF_LIBRARIES})
target_include_directories(wave260uni PUBLIC ${MPI_INCLUDE_PATH} ${netCDF_INCLUDE_PATH})

set(CMAKE_VERBOSE_MAKEFILE ON)
set(BUILD_SHARED_LIBS OFF) 
target_compile_features(wave260mpi PUBLIC cxx_std_17)
target_compile_features(wave260uni PUBLIC cxx_std_17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
