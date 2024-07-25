import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Descriptors, MolFromSmiles, AddHs, MolToSmiles, MolFromSmarts
from rdkit.Chem.EnumerateStereoisomers import (
    GetStereoisomerCount,
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from math import log
from joblib import Parallel, delayed
import itertools

import os
from htp_docking.docking_openeye import *

RDLogger.DisableLog("rdApp.*")

#_base_dir = os.path.split(__file__)[0]
##_mcf = pd.read_csv(os.path.join(_base_dir, "mcf.csv"))
#_filters = [MolFromSmarts(sm) for sm in _mcf["smarts"].tolist()]
#_counts = _mcf["counts"].tolist()

def filter_dock(
    smiles_list,
    receptor_oedu_file='receptor',
    max_confs = 1,
    score_cutoff=-9,
    temp_dir='tmp',
    n_jobs=1,
    debug=False,
):

    """
    Perform filtering of a list of smiles, optionally in parallel

    Filters currently implemented:
        - check if smiles is valid
        - check if smiles contain substruct match with given scaffold
        - check if rings < 8
        - check if molecular weight in [300, 600]
        - check if # chiral centers < 2
        - check if smiles contains any of the manually-curated smarts filters (contained in _mcf.csv)

    Args:
        scaffold (optional): smarts string representing scaffold
        n_jobs (int, default=1): number of threads to use to perform filtering in parallel
    """
    def filt(smiles, receptor_file, max_confs, temp_dir):
        try:
            os.mkdir(out_lig_dir)
        except:
            pass
        try:
            os.mkdir(temp_dir)
        except:
            pass
        passes = []
        receptor = read_receptor(receptor_file)
        #if scaff is not None:
        #    scaff = MolFromSmarts(scaff)
        for it, smi in enumerate(smiles):
            try:
                dock_score = run_docking_score(
                                smi,
                                it,
                                receptor,
                                max_confs,
                                temp_dir,
                                )
                print(f"Docking score is: {dock_score}")
                #print(it)
                print(smi)
                if dock_score > score_cutoff: 
                    passes.append(False)
                    continue
            except:
                print("exception")
                passes.append(False)
                continue
            passes.append(True)
            print(passes)
        return passes

    if n_jobs == 1:
        return filt(smiles_list, receptor_oedu_file, max_confs, temp_dir)
    else:
        n_jobs = int(n_jobs)
        assert n_jobs > 1
        out = Parallel(n_jobs=n_jobs, max_nbytes=None, prefer="threads")(
            delayed(filt)(b, receptor_oedu_file, max_confs, temp_dir) for b in np.array_split(smiles_list, n_jobs)
        )
        return list(itertools.chain.from_iterable(out))
