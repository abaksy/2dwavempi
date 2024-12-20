#/usr/bin/python
import os
import sys
import argparse
import math

CORES_PER_NODE = 128
parser = argparse.ArgumentParser()
# required args
parser.add_argument("test", help='JSON config file')
# parser.add_argument("-x", type=int, required=True, help='processor geometry columns')
# parser.add_argument("-y", type=int, required=True, help='processor geometry row')



# optional args
parser.add_argument("-g", default=None, help= 'comma separated list of one or more geometries specified as rxc where there are r rows of processors by c columns of processors.  For example -g 4x4,8x2,2x8. If not specified, run the uniprocessor binary.' )
parser.add_argument("-n", type=int, default=None, help='matrix size')
parser.add_argument("-i", type=int, default=None, help='number of iterations')
parser.add_argument("-p", type=int, default=None, help='plot interval')
parser.add_argument("-k", action="store_true", default=False, help='disable communication')
args = parser.parse_args()

if (args.g == None):
    wavebin = "wave260uni"
    np0=1
    geolist=["1x1"]
else:
    wavebin = "wave260mpi"
    geolist = args.g.split(',')

    (py0, px0) = geolist[0].split('x')
    np0 = int(py0) * int(px0)
    for g in geolist:
        (px, py) = g.split('x')
        np = int(px) * int(py)
        if (np != np0):
            print("Error py ({}) * px ({}) = {} which is not equal to {} * {} n ({})".format(py,
                                                                                         px,
                                                                                         np,
                                                                                         py0,        
                                                                                         px0,
                                                                                         np0))
            sys.exit()

if args.n == None:
    nsize = ""
else:
    nsize = "-n " + str(args.n)

if args.i == None:
    iter = ""
else:
    iter = "-i " + str(args.i)

if args.p == None:
    plot = ""
    plotopt = ""
else:
    plot = "-p " + str(args.p)
    plotopt = "_p" + str(args.p)

if args.k:
    nocomm = "-k"
    nocommopt = "_k"
else:
    nocomm = ""
    nocommopt = ""


    
testfile = args.test
if (not os.path.exists(testfile)):
    print("Error: can't open {}".format(testfile)) 
    sys.exit()


testroot = os.path.basename(testfile)

testname_with_options = os.path.splitext(testroot)[0] + "_n" + str(np0) + plotopt + nocommopt


    
cores = np0
nodes = math.ceil(cores/CORES_PER_NODE)
cores_per_node = int(cores / nodes)
if (cores % nodes) != 0:
    print("Error: {} nodes cannot evenly divide {} cores".format(nodes, cores))
    sys.exit()
    
if (cores_per_node == 0):
    cores_per_node = 128

if (cores <= 32):
    partition = "shared"
else:
    partition = "compute"



#
# give approv .75 GB/core
# really depends on the size of the grid but
# since the info is in the .config file, we don't know it here
#
mem_limit = int(0.55 * cores_per_node)

user = os.getlogin()



#
# create output file
#
ofilename = testname_with_options + ".slurm"
print("Creating slurm script {}".format(ofilename))
f = open(ofilename, "w")    


###### template slrum script ######
text = """#!/bin/bash
#SBATCH --job-name="../wave260"
#SBATCH --output="{testname_with_options}.%j.%N.out"
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cores_per_node}
#SBATCH --mem={mem_limit}G
#SBATCH --account=csd911
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
export MV2_USE_RDMA_CM=0
export MV2_HOMOGENEOUS_CLUSTER=1

module purge
module load cpu/0.17.3b  gcc/10.2.0/npcyll4  mvapich2/2.3.7/iyjtn3x
module load cmake/3.21.4/n5jtjsf
module load netcdf-c/4.8.1/yt65vte
module load ncview/2.1.8/znk6ds3
module load tau/2.30.2/auogdal
module load slurm

mkdir -p /expanse/lustre/scratch/{user}/temp_project/job_$SLURM_JOB_ID
"""



f.write(text.format(testname_with_options = testname_with_options,
                  partition=partition,
                  nodes=nodes,
                  cores_per_node=cores_per_node,
                  mem_limit=mem_limit,
                  user=user,
                  cores=cores,
                  plot=plot,
                  iter=iter,
                  nocomm=nocomm,
                  testfile=testfile))
py=1
px=1
for g in geolist:
    (py,px) = g.split('x')
    text = """
srun --chdir /expanse/lustre/scratch/{user}/temp_project/job_$SLURM_JOB_ID --mpi=pmi2 -n {cores} $SLURM_SUBMIT_DIR/../build_expanse/{wavebin} $SLURM_SUBMIT_DIR/../tests/{testfile}  -y {py} -x {px} {nsize} {iter} {plot} {nocomm}
    """
    f.write(text.format(user=user,
                        cores=cores,
                        wavebin=wavebin,
                        testfile=testfile,
                        py=py,
                        px=px,
                        nsize=nsize,
                        iter=iter,
                        plot=plot,
                        nocomm=nocomm))


f.close()

