import numpy as np
import h5py
import run
import sys
from tqdm import tqdm
import sqlite3


if __name__ == '__main__':
    dataset = sys.argv[1]
    f = h5py.File(run.DATASETS[dataset](), 'r+')
    # Transform similarities into distances
    sims = f['top-1000-pairs'][:, 0]
    assert (sims >= 0).all()
    distances = 1.0 - sims
    sample = f['sample-similarities'][:]
    avg_dist = (1.0 - sample).mean()
    print('average distance is ', avg_dist)
    contrasts = avg_dist / distances
    
    with run.get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS pair_rc (
            dataset          TEXT,
            pair_rank        INT,
            pair_similarity  REAL,
            pair_rc          REAL,
            PRIMARY KEY (dataset, pair_rank)
        )
        """)

        dbdata = [
            (dataset, rank, sim, rc)
            for rank, (sim, rc) in enumerate(zip(sims, contrasts))
        ]
        db.executemany("INSERT INTO pair_rc VALUES(?, ?, ?, ?)", dbdata)


        # db.execute("""
        # INSERT INTO dimensionality (dataset, k, lid_mean, lid_median, lid_min, lid_max, lid_25, lid_75)
        # VALUES (:dataset, :k, :lid_mean, :lid_median, :lid_min, :lid_max, :lid_25, :lid_75)
        # """, {
        #     'dataset': dataset,
        #     'k': k,
        #     'lid_mean': np.mean(lids),
        #     'lid_median': np.median(lids),
        #     'lid_min': np.min(lids),
        #     'lid_max': np.max(lids),
        #     'lid_25': np.quantile(lids, 0.25),
        #     'lid_75': np.quantile(lids, 0.75)
        # })

