#!/usr/bin/env python3

# SBATCH --job-name=particle_tracking
# SBATCH --partition=high2    # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --mem=62G    # need to specify memory if you set the number of tasks (--ntasks) below
# SBATCH --nodes=1    # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1    # (MPI) tasks per node
# SBATCH --ntasks=1    # (MPI) tasks total
##SBATCH --cpus-per-task=1    # (OpenMP) threads per (MPI) task
# SBATCH --time=00:30:00
# SBATCH --output=particle_tracking_%j.txt
# SBATCH --mail-user=your_email@ucdavis.edu
# SBATCH --mail-type=fail
# SBATCH --mail-type=end

import os
import utilities.io as ut_io  # if you want to use my print diagnostics
import utilities as ut
import gizmo_analysis as gizmo
from trackingCode import *

# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass("slurm")

simulation_directory = "/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500"

z0 = gizmo.io.Read.read_snapshots(
    simulation_directory=simulation_directory,
    species=["star"],
    snapshot_values=600,
    snapshot_value_kind="index",
    assign_hosts_rotation=True,
    assign_formation_coordinates=True,
)

indices = ut.array.get_indices(z0["star"].prop("host2.distance.total"), [6, 6.5])
property_list = ["host2.distance.principal.cylindrical", "mass"]
snapshots = np.arange(1, 601, 1)

runParallelStellarPropertyTracking(
    simulation_directory,
    indices,
    property_list,
    snapshots=snapshots,
    proc_number=1,
    float32=True,
)

cleanup(snapshots=snapshots)
