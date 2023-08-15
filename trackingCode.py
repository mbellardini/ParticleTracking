"""
@author:
    Matt Bellardini <mbellardini@ucdavis.edu>
"""

# Import programs
import numpy as np
import utilities.io as ut_io
import pickle
import gizmo_analysis as gizmo
import os
import glob


def shortFormSim(sim, secondary_host=False):
    """
    Quick and dirty code to shorten names of simulations for temporary files.
    e.g. /m12b/m12b_r7100 -> m12b

    Parameters
    ----------
    sim: str
      - Long version of the simulation name to be shortened
    secondary_host: bool
      - Whether to return the name of the less massive host for elvis sims

    Returns
    -------
    string: abbreviated host name
    """

    if "elvis" not in sim:
        idx = sim.find("m12")
        return sim[idx : idx + 4]
    else:
        for host in ["Romeo", "Romulus", "Thelma", "Juliet", "Remus", "Louise"]:
            if host in sim:
                if secondary_host:
                    if host == "Romeo":
                        return "Juliet"
                    elif host == "Romulus":
                        return "Remus"
                    else:  # host == 'Thelma'
                        return "Louise"
                else:
                    return host

        else:
            raise NameError(f"host not recognized for {sim}")


def trackStellarProperties(
    simulation_directory, short_name, index, prior_indices, property_list, float32=True
):
    """
    Used to track properties of individual star particles across snapshots. Saves
    a temporary file for each snapshot.

    Parameters
    ----------
    simulation_directory: str
      - The path to the directory where the snapshots being read in are located
    index: int
      - The index of the snapshot being read in
    prior_indices: [int]
      - The indices of the star particles at z=0 that are to be tracked
    property_list: [str]
      - A list of properties to be tracked across snapshots, these must all be
      compatable with the gizmo_io.ParticleDictionaryClass.prop() method
    float32: bool
      - Whether to store variables as 32 bit floats rather than 64
    """

    m12 = gizmo.io.Read.read_snapshots(
        simulation_directory=simulation_directory,
        species=["star"],
        snapshot_values=index,
        snapshot_value_kind="index",
        assign_hosts_rotation=True,
        assign_pointers=True,
    )

    pointers = m12.Pointer.get_pointers(
        species_name_from="star", species_names_to="star"
    )
    indices = pointers[prior_indices]
    idxa = np.where(indices >= 0)
    idxb = np.where(indices < 0)
    pos_indices = indices[idxa]
    neg_indices = indices[idxb]

    output = {}
    for prop in property_list:
        p = m12["star"].prop(prop)
        arr = (
            np.nan * np.empty(len(indices))
            if len(p.shape) == 1
            else np.nan * np.empty([len(indices), p.shape[1]])
        )
        arr[idxa] = p[pos_indices]
        output[prop] = np.float32(arr) if float32 else arr

    if index < 10:
        index = f"00{index}"
    elif index < 100:
        index = f"0{index}"

    ut_io.file_hdf5(
        f"temporary_stellar_tracking/{short_name}_{index}",
        dict_or_array_to_write=output,
        verbose=True,
    )


def runParallelStellarPropertyTracking(
    simulation_directory,
    indices,
    property_list,
    snapshots=np.arange(1, 600, 1),
    proc_number=2,
    float32=True,
):
    """
    Runs trackStellarProperties in parallel. Saves temporary files in a directory,
    which will be created if it does not already exist. Temporary files can be merged
    by running the cleanup script.

    Parameters
    ----------
    simulation_directory: str
      - The path to the directory where the snapshots being read in are located
    indices: [int]
      - The indices of the star particles at z=0 that are to be tracked
    property_list: [str]
      - A list of properties to be tracked across snapshots, these must all be
      compatable with the gizmo_io.ParticleDictionaryClass.prop() method
    snapshots: [int]
      - A list of snapshot indices over which the particles are tracked
    proc_number: int
      - Number of parallel processes to run
    float32: bool
      - Whether to store variables as 32 bit floats rather than 64
    """

    if not os.path.exists("temporary_stellar_tracking"):
        os.mkdir("temporary_stellar_tracking")

    z0 = gizmo.io.Read.read_snapshots(
        simulation_directory=simulation_directory,
        species=["star"],
        snapshot_values=600,
        snapshot_value_kind="index",
        assign_hosts_rotation=True,
        assign_formation_coordinates=True,
    )

    output = {}
    secondary = False
    for prop in property_list:
        p = z0["star"].prop(prop, indices=indices)
        output[prop] = np.float32(p) if float32 else p
        if "host2" in prop:
            secondary = True

    short_name = shortFormSim(simulation_directory, secondary)
    ut_io.file_hdf5(
        f"temporary_stellar_tracking/{short_name}_600",
        dict_or_array_to_write=output,
        verbose=True,
    )

    # Free up some memory for the parallel computing
    del output
    del z0

    args = [
        [simulation_directory, short_name, snapshot, indices, property_list]
        for snapshot in snapshots
    ]

    ut_io.run_in_parallel(
        func=trackStellarProperties,
        args_list=args,
        proc_number=proc_number,
        verbose=True,
    )


def cleanup(snapshots=np.arange(1, 601, 1)):
    """
    For use after running runParallelStellarPropertyTracking. Reads in all
    temporary files and combines them into one large file. Not necessary to use
    unless all data needs to be worked with at once.
    """

    if not os.path.exists("final_stellar_tracking"):
        os.mkdir("final_stellar_tracking")

    arr = glob.glob("temporary_stellar_tracking/*")
    short_names = []
    for temp in arr:
        name = shortFormSim(temp)
        if name not in short_names:
            short_names.append(name)
    del arr

    for name in short_names:
        file_names = glob.glob(f"temporary_stellar_tracking/{name}*")
        output = {}
        for f in file_names:
            temp = ut_io.file_hdf5(f)
            if not output:
                for prop in temp:
                    shape = [s for s in temp[prop].shape]
                    shape.insert(0, snapshots[-1] - snapshots[0] + 1)
                    output[prop] = np.nan * np.empty(shape, temp[prop].dtype)
            for prop in temp:
                idx = int(f[-8:-5]) - snapshots[0]
                output[prop][idx] = temp[prop]

        output["snapshots"] = snapshots
        ut_io.file_hdf5(f"final_stellar_tracking/{name}_particle_tracking", output)
