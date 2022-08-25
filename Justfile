set dotenv-load

build:
  cmake --build build --config Debug --target PuffinnJoin

check:
  #!/bin/bash
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  env OMP_NUM_THREADS=56 time build/PuffinnJoin < instructions.txt > result.dsv

cache-misses exe:
  perf record --call-graph dwarf -e cache-misses -p $(pgrep {{exe}})

profile exec:
  perf record --call-graph dwarf -p $(pgrep {{exec}})

# install flamegraph
install-flamegraph:
  # Check and install rust
  cargo --version || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  # Check and install flamegraph
  flamegraph --version || cargo install flamegraph

test_lsh_join:
  cmake --build build --config Debug --target Test
  env OMP_NUM_THREADS=1 build/Test Index::lsh_join

bench:
  cmake --build build --config RelWithDebInfo --target Bench
  # env OMP_NUM_THREADS=56 build/Bench /mnt/large_storage/topk-join/datasets/orkut.hdf5 # >> bench_results.txt
  env OMP_NUM_THREADS=56 build/Bench /mnt/large_storage/topk-join/datasets/glove-200.hdf5 

# open the sqlite result database
sqlite:
  sqlite3 $TOPK_DIR/join-results.db

run:
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  cmake --build build --config RelWithDebInfo --target LSBTree
  cmake --build build --config RelWithDebInfo --target XiaoEtAl
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/run.py

lid dataset:
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/lid.py {{dataset}}

plot:
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/plot.py

distr data:
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/plot.py {{data}}

console:
  cd join-experiments && env TOPK_DIR=/mnt/large_storage/topk-join/ python3

bash:
  cd /mnt/large_storage/topk-join/ && bash

compute-recalls:
  cd join-experiments && env TOPK_DIR=/mnt/large_storage/topk-join/ python3 -c 'import run; run.compute_recalls(run.get_db())'

