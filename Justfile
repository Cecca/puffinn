set dotenv-load

check:
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  build/PuffinnJoin < instructions.txt > result.dsv

cache-misses exe:
  sudo perf record -e cache-misses -p $(pgrep {{exe}})

# produce a flamegraph.svg file
profile exec: install-flamegraph
  flamegraph --root -p $(pgrep {{exec}})

# install flamegraph
install-flamegraph:
  # Check and install rust
  cargo --version || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  # Check and install flamegraph
  flamegraph --version || cargo install flamegraph

test:
  cmake --build build --config RelWithDebInfo --target Test
  env OMP_NUM_THREADS=56 build/Test

bench:
  cmake --build build --config RelWithDebInfo --target Bench
  env OMP_NUM_THREADS=56 build/Bench datasets/glove-200-100k.hdf5 >> bench_results.txt
  cat bench_results.txt

# open the sqlite result database
sqlite:
  sqlite3 $TOPK_DIR/join-results.db

run:
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  cmake --build build --config RelWithDebInfo --target LSBTree
  cmake --build build --config RelWithDebInfo --target XiaoEtAl
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/run.py

plot:
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/plot.py
