"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
#import MDAnalysis as mda
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import List, Optional
import numpy as np
from openeye import oechem, oedocking, oeomega
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm
from htp_docking.docking_utils import smi_to_structure
from htp_docking.utils import exception_handler
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

'''
Functions
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
import pickle
from mpi4py import MPI
import random

#data_pwd ='/lus/eagle/projects/datascience/avasan/Workflow/EarlyESPRuns/Data/InferenceData'
#dataset = 'ZIN'
#sample_size = 1000000
random_state = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sample_dataset(data_pwd, dataset, size, rank, sample_size):
    list_dir_files = np.array(sorted(os.listdir(f'{data_pwd}/{dataset}')))
    num_files = len(list_dir_files)
    list_dir_files = np.array_split(list_dir_files, size)[rank]
    n = int(sample_size/num_files)+1#sample_size_per_file
    sample_dataset = []#np.empty(sample_size, dtype='str')#[]
    sample_perrank = sample_size/size
    for fil in tqdm(list_dir_files):
        with open(f'{data_pwd}/{dataset}/{fil}', 'r') as fil:
            data = fil.read().splitlines()[1:]
            del(fil)
        sample_fil = sample(data, n)
        sample_dataset.extend(sample_fil)
        del(sample_fil)
        del(data)
        if len(sample_dataset)>=sample_perrank:
            return sample_dataset
    return sample_dataset

def smi_to_structure(smiles: str, output_file: Path, forcefield: str = "mmff") -> None:
    """Convert a SMILES file to a structure file.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output_file : Path
        EIther an output PDB file or output SDF file.
    forcefield : str, optional
        Forcefield to use for 3D conformation generation
        (either "mmff" or "etkdg"), by default "mmff".
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate a 3D conformation for the molecule
    if forcefield == "mmff":
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "etkdg":
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    else:
        raise ValueError(f"Unknown forcefield: {forcefield}")

    # Write the molecule to a file
    if output_file.suffix == ".pdb":
        writer = Chem.PDBWriter(str(output_file))
    elif output_file.suffix == ".sdf":
        writer = Chem.SDWriter(str(output_file))
    else:
        raise ValueError(f"Invalid output file extension: {output_file}")
    writer.write(mol)
    writer.close()



def from_mol(mol, isomer=True, num_enantiomers=1):
    """
    Generates a set of conformers as an OEMol object
    Inputs:
        mol is an OEMol
        isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
        num_enantiomers is the allowable number of enantiomers. For all, set to -1
    """
    # Turn off the GPU for omega
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    omega = oeomega.OEOmega(omegaOpts)

    out_conf = []
    if not isomer:
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            out_conf.append(mol)
        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    elif isomer:
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(enantiomer)
                num_enantiomers -= 1
                if num_enantiomers == 0:
                    break
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    return out_conf


def from_string(smiles, isomer=True, num_enantiomers=1):
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"SMILES invalid for string {smiles}")
    else:
        return from_mol(mol, isomer, num_enantiomers)


def from_structure(structure_file: Path) -> oechem.OEMol:
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    ifs = oechem.oemolistream()
    if not ifs.open(str(structure_file)):
        raise ValueError(f"Could not open structure file: {structure_file}")

    if structure_file.suffix == ".pdb":
        oechem.OEReadPDBFile(ifs, mol)
    elif structure_file.suffix == ".sdf":
        oechem.OEReadMDLFile(ifs, mol)
    else:
        raise ValueError(f"Invalid structure file extension: {structure_file}")

    return mol


def select_enantiomer(mol_list):
    return mol_list[0]


def dock_conf(receptor, mol, max_poses: int = 1):
    dock = oedocking.OEDock()
    dock.Initialize(receptor)
    lig = oechem.OEMol()
    err = dock.DockMultiConformerMolecule(lig, mol, max_poses)
    return dock, lig

# Returns an array of length max_poses from above. This is the range of scores
def ligand_scores(dock, lig):
    return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]


def best_dock_score(dock, lig):
    return ligand_scores(dock, lig)#[0]


def write_ligand(ligand, output_dir: Path, smiles: str, lig_identify: str) -> None:
    # TODO: If MAX_POSES != 1, we should select the top pose to save
    ofs = oechem.oemolostream()
    for it, conf in enumerate(list(ligand.GetConfs())):
        if ofs.open(f'{str(output_dir)}/{lig_identify}/{it}.pdb'):
            oechem.OEWriteMolecule(ofs, conf)
            ofs.close()
    return
    raise ValueError(f"Could not write ligand to {output_path}")


def write_receptor(receptor, output_path: Path) -> None:
    ofs = oechem.oemolostream()
    if ofs.open(str(output_path)):
        mol = oechem.OEMol()
        contents = receptor.GetComponents(mol)#Within
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    return
    raise ValueError(f"Could not write receptor to {output_path}")


#@cache  # Only read the receptor once
def read_receptor(receptor_oedu_file: Path):
    """Read the .oedu file into a GraphMol object."""
    receptor = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
    return receptor

#@cache
def create_proteinuniv(protein_pdb):
    protein_universe = mda.Universe(protein_pdb)
    return protein_universe

def create_complex(protein_universe, ligand_pdb):
    u1 = protein_universe
    u2 = mda.Universe(ligand_pdb)
    u = mda.core.universe.Merge(u1.select_atoms("all"), u2.atoms)#, u3.atoms)
    return u

def create_trajectory(protein_universe, ligand_dir, output_pdb_name, output_dcd_name):
    import MDAnalysis as mda
    ligand_files = sorted(os.listdir(ligand_dir))
    comb_univ_1 = create_complex(protein_universe, f'{ligand_dir}/{ligand_files[0]}').select_atoms("all")

    with mda.Writer(output_pdb_name, comb_univ_1.n_atoms) as w:
        w.write(comb_univ_1)
    with mda.Writer(output_dcd_name, comb_univ_1.n_atoms,) as w:
        for it, ligand_file in enumerate(ligand_files):
            comb_univ = create_complex(protein_universe, f'{ligand_dir}/{ligand_file}') 
            w.write(comb_univ)    # write a whole universe
            os.remove(f'{ligand_dir}/{ligand_file}')
    return

@exception_handler(default_return=0.0)
def run_docking(
    smiles: str, 
    lig_identify: str,
    receptor_oedu_file: Path,
    max_confs: int,
    score_cutoff: float,
    protein_pdb: Path,
    temp_dir: Path,
    out_lig_dir: Path,
    temp_storage: Optional[Path] = None
) -> float:
    """Run OpenEye docking on a single ligand.

    Parameters
    ----------
    smiles : ste
        A single SMILES string.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    max_confs : int
        Number of ligand poses to generate
    temp_lig_dir : Path
        Temporary directory to write individual ligand poses
    out_lig_path : Path
        Location to output ligand_protein top n poses --> single pdb
    out_rec_path : Path
        Where to write output receptor file
    temp_storage : Path
        Path to the temporary storage directory to write structures to,
        if None, use the current working Python's built in temp storage.

    Returns
    -------
    float
        The docking score of the best conformer.
    """

    try:
        conformers = select_enantiomer(from_string(smiles))
    except:
        with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
            # Read input SMILES and generate conformer
            smi_to_structure(smiles, Path(fd.name))
            conformers = from_structure(Path(fd.name))

    # Read the receptor to dock to
    receptor = read_receptor(receptor_oedu_file)

    # Dock the ligand conformers to the receptor
    dock, lig = dock_conf(receptor, conformers, max_poses=max_confs)

    # Get the docking scores
    best_score = best_dock_score(dock, lig)

    # If receptor pdb file doesnt exist then create one
    if False:
        out_rec_path = f'{temp_dir}/{Path(receptor_oedu_file).stem}.pdb'
        if not os.path.isfile(out_rec_path):
            write_receptor(receptor, out_rec_path)

    if best_score[0]<score_cutoff:
        sys.stdout.flush()
        if not os.path.isdir(f'{temp_dir}/{lig_identify}'):
            os.mkdir(f'{temp_dir}/{lig_identify}')
        write_ligand(lig, temp_dir, smiles, lig_identify)
        protein_universe = create_proteinuniv(protein_pdb)
        create_trajectory(protein_universe, f'{temp_dir}/{lig_identify}', f'{out_lig_dir}/pdbs/{Path(receptor_oedu_file).stem}.{lig_identify}.pdb',f'{out_lig_dir}/dcds/{Path(receptor_oedu_file).stem}.{lig_identify}.dcd')
    return np.mean(best_score[0])

@exception_handler(default_return=0.0)
def run_docking_score(
    smiles: str, 
    lig_identify: str,
    receptor,
    max_confs: int,
    temp_storage: Optional[Path] = None
) -> float:
    """Run OpenEye docking on a single ligand.

    Parameters
    ----------
    smiles : ste
        A single SMILES string.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    max_confs : int
        Number of ligand poses to generate
    temp_lig_dir : Path
        Temporary directory to write individual ligand poses
    out_lig_path : Path
        Location to output ligand_protein top n poses --> single pdb
    out_rec_path : Path
        Where to write output receptor file
    temp_storage : Path
        Path to the temporary storage directory to write structures to,
        if None, use the current working Python's built in temp storage.

    Returns
    -------
    float
        The docking score of the best conformer.
    """

    try:
        conformers = select_enantiomer(from_string(smiles))
    except:
        with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
            # Read input SMILES and generate conformer
            smi_to_structure(smiles, Path(fd.name))
            conformers = from_structure(Path(fd.name))

    # Dock the ligand conformers to the receptor
    dock, lig = dock_conf(receptor, conformers, max_poses=max_confs)

    # Get the docking scores
    best_score = best_dock_score(dock, lig)
    return np.mean(best_score)

def run_mpi_docking(
    rank,
    size,
    data_pwd,
    dataset,
    receptor_oedu_file: Path,
    max_confs: int,
    out_csv_pattern: str,
    batch_size: int,
    pose_gen: bool,
    sample_size: Optional[int] = 1000,
    seed: Optional[int]=50,
    score_cutoff: Optional[float] = 10,
    protein_pdb: Optional[Path] = 'protein.pdb',
    temp_dir: Optional[Path] = 'temp_dir',
    out_lig_dir: Optional[Path] = 'output_traj',
    temp_storage: Optional[Path] = None) -> float:
    
    smiles_list_new = sample_dataset(data_pwd, dataset, size, rank, sample_size)
    # Read the receptor to dock to
    receptor = read_receptor(receptor_oedu_file)
    smiles_index_new = [ind for ind in range(len(smiles_list_new))]
    smiles_list_batches = smiles_list_new #np.array_split(smiles_list_new, int(len(smiles_list_new)//batch_size))
    smiles_index_batches = smiles_index_new #np.array_split(smiles_index_new, int(len(smiles_index_new)//batch_size))

    if not pose_gen:
        #for batch_it, (batch_s, batch_i) in tqdm(enumerate(zip(smiles_list_batches, smiles_index_batches))):
        scores = [run_docking_score(smi, f'{smi_ind}', receptor, max_confs, temp_dir) for smi, smi_ind in tqdm(zip(smiles_list_batches, smiles_index_batches))]
        df_new = pd.DataFrame({'SMILES':smiles_list_batches,'Scores':scores})
        df_new.to_csv(f'{out_csv_pattern}.{rank}.csv', index=False)

    if pose_gen:
            scores = [run_docking(smi, f'{smiles_index_new[it]}', receptor_oedu_file, max_confs, score_cutoff, protein_pdb, temp_dir, out_lig_dir) for it, smi in enumerate(smiles_list_new)]
    return 1

'''
Running Code
'''

if False:
    ### Initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    size = comm.Get_size()
    
    ### Parameters setting
    import json
     
    # Opening JSON file
    f = open('config_htp.json')
     
    # returns JSON object as 
    # a dictionary
    config = json.load(f)
    
    data_pwd = config['data_pwd']
    dataset = config['dataset']
    recep_file = config['recep_file'] # receptor oedu file for openeye 
    max_confs = config['max_confs'] # confs to generate
    pose_gen = config['pose_gen'] # Boolean... should we gen poses?
    batch_size = config['batch_size']
    score_cutoff = config['score_cutoff'] # below this score generate pose
    temp_dir = config['temp_dir']#'lig_confs' # store ligand poses temporarily
    output_traj = config['output_traj']#'output_combined_trajectories' # store pdb + dcd files for complex poses
    score_pattern = config['score_pattern']#'scores/4ui5_scores' #store scores like this (will have #ranks files)
    protein_pdb = config['protein_pdb'] #protein pdb file to use to store. Will save everything in this file to complex
    sample_size = config['sample_size'] 
    score_output = config['score_output']
    train_out = config['train_out']
    
    run_mpi_docking(rank,
                    size,
                    data_pwd,
                    dataset,
                    recep_file,
                    max_confs,
                    score_pattern,
                    batch_size,
                    pose_gen,
                    sample_size = sample_size,
                    temp_dir = temp_dir,
                    temp_storage = temp_dir)
    
    #'''
    #Set a comm barrier and merge all files.
    #'''
    #
    #comm.Barrier()
    #if rank==10:
    #    fil_list = sorted(os.listdir(f'{score_output}'))
    #
    #    df = pd.read_csv(f'{score_output}/{fil_list[0]}')
    #    
    #    for i in range(1, len(fil_list)):
    #        df = pd.concat([df, pd.read_csv(f'{score_output}/{fil_list[i]}')])
    #    
    #    df.to_csv(f'{train_out}/unprocessed.csv', index=False)
    
    
    
    
    '''
    combining all score files into one. For now just do this outside of the script
    '''
    if False:
        comm.Barrier()
        if rank==0:
            df = pd.read_csv(f'{score_pattern}.0.csv')
            for r in range(1, size):
                df = pd.concat([df, pd.read_csv(f'{score_pattern}.{r}.csv')])
            df.to_csv(f'{score_pattern}.csv')
    
