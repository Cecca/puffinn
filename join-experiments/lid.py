import numpy as np
import h5py
import run
import sys
from tqdm import tqdm

# Adapted from https://github.com/Cecca/role-of-dimensionality/blob/is_revision/additional-scripts/compute-lid.py
def lid(dists, k):
    dists.sort()
    w = dists[min(len(dists) - 1, k)]
    half_w = 0.5 * w

    dists = dists[:k+1]
    dists = dists[dists > 1e-5]

    # Use numpy vector operations to improve efficiency.
    # Results are the same up the the 6th decimal position
    # compared to iteration

    small = dists[dists < half_w]
    large = dists[dists >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size
    return -valid / s


def all_lid(all_dists, k):
    return np.array([
        lid(all_dists[i,:],k)
        for i in tqdm(range(all_dists.shape[0]))
    ])


if __name__ == '__main__':
    dataset = sys.argv[1]
    f = h5py.File(run.DATASETS[dataset](), 'r+')
    # Transform similarities into distances
    sims = f['top-1000-dists'][:]
    assert (sims >= 0).all()
    distances = 1.0 - sims
    if 'LID-10' not in f:
        print("LID @ 10")
        f['LID-10'] = all_lid(distances, 10)
    if 'LID-100' not in f:
        print("LID @ 100")
        f['LID-100'] = all_lid(distances, 100)
    

