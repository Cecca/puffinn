#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <chrono>
#include <omp.h>
#include "highfive/H5Easy.hpp"
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

const unsigned long long MB = 1024*1024;
const unsigned long long KB = 1024;

class Progress {
  std::string prefix;
  std::chrono::time_point<std::chrono::system_clock> start;
  uint64_t every;
  uint64_t last_log;
  uint64_t count;
  uint64_t expected;

  void log() const {
    auto cur = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(cur - start).count();
    double rate = count / elapsed;
    uint64_t todo = expected - count;
    double estimate = todo / rate;
    std::cerr << prefix << count << " elapsed " << elapsed << " :: " << estimate << " to go" << std::endl;
  }

public:
  Progress(std::string prefix, uint64_t every, uint64_t expected): prefix(prefix), start(std::chrono::system_clock::now()), every(every), last_log(0), count(0), expected(expected) {}

  void update(uint64_t x) {
    #pragma omp atomic
    count += x;
    if (count - last_log >= every) {
        #pragma omp critical 
        {
            log();
            last_log = count;
        }
    }
  }
};

struct Pair
{
  size_t a;
  size_t b;
  float similarity;
};

// Returns true if pair a is < than pair b, i.e. if a.similarity > b.similarity
bool cmp_pairs(const Pair &a, const Pair &b)
{
  return a.similarity > b.similarity;
}

float dotp(std::vector<float> &a, std::vector<float> &b) {
  size_t d = a.size();
  float p = 0.0;
  for (size_t i=0; i<d; i++) {
    p += a[i] * b[i];
  }
  return p;
}

float euclidean(std::vector<float> &a, std::vector<float> &b) {
  size_t d = a.size();
  float s = 0.0;
  for (size_t i=0; i<d; i++) {
    float diff = (a[i] - b[i]);
    s += diff * diff;
  }
  return std::sqrt(s);
}

float norm(std::vector<float> & v) {
    float n = 0.0;
    for (auto x : v) {
        n += x * x;
    }
    return std::sqrt(n);
}


int main(int argc, char** argv) {
    std::string path;
    size_t k = 1000;

    if (argc == 2) {
        path = argv[1];
    } else if (argc == 4) {
        assert(argv[1] == "-k");
        k = atoi(argv[2]);
        path = argv[3];
    } else {
        std::cerr << "USAGE: TopPairsCosine [-k K] <dataset>" << std::endl;
        return 1;
    }
    bool exact = false;

    std::cerr << "loading data from " << path << std::endl;
    H5Easy::File file(path, H5Easy::File::ReadWrite);
    std::vector<std::vector<float>> data = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
    size_t n = data.size();
    size_t dim = data[0].size();
    for (size_t i=0; i<data.size(); i++) {
        float n = norm(data[i]);
        for (size_t j=0; j<data[i].size(); j++) {
            data[i][j] /= n;
        }
    }
    std::cerr << "loaded " << n
              << " vectors from hdf5 file, of dimension "
              << dim << std::endl;

    std::vector<std::vector<float>> out_res;

    if (exact) {
        std::vector<std::vector<Pair>> threads_res(omp_get_max_threads());

        Progress prog("point ", 1000, data.size());
        #pragma omp parallel for schedule(dynamic)
        for (size_t i=0; i<data.size(); i++) {
            std::vector<Pair> & res = threads_res[omp_get_thread_num()];
            for (size_t j=i+1; j<data.size(); j++) {
                float sim = dotp(data[i], data[j]);
                if (sim > 1.0) {
                    sim = 1.0;
                }
                res.push_back(Pair{i, j, sim});
                std::push_heap(res.begin(), res.end(), cmp_pairs);
                if (res.size() > k) {
                    std::pop_heap(res.begin(), res.end(), cmp_pairs);
                    res.pop_back();
                }
            }
            prog.update(1);
        }

        std::vector<Pair> res = threads_res[0];
        for (size_t tid=1; tid < threads_res.size(); tid++) {
            for (auto pair : threads_res[tid]) {
                res.push_back(pair);
                std::push_heap(res.begin(), res.end(), cmp_pairs);
                if (res.size() > k) {
                    std::pop_heap(res.begin(), res.end(), cmp_pairs);
                    res.pop_back();
                }
            }
        }

        std::sort_heap(res.begin(), res.end(), cmp_pairs);
        for (auto pair : res) {
            out_res.push_back({pair.similarity, pair.a, pair.b});
        }
    } else {
        puffinn::Index<puffinn::CosineSimilarity, puffinn::SimHash> index(
            data[0].size(),
            4 * KB * data.size(),
            puffinn::IndependentHashArgs<puffinn::SimHash>()
        );
        for (auto v : data) { index.insert(v); }
        index.rebuild(false, false);
        auto pairs = index.global_lsh_join(k, 0.999);
        for (auto entry : pairs.best_entries()) {
            std::vector<float> vpair;
            vpair.push_back(entry.second);
            vpair.push_back(entry.first.first);
            vpair.push_back(entry.first.second);
            out_res.push_back(vpair);
        }

    }

    /* for (size_t i=res.size(); i>res.size() - 10; i--) { */
    /*     std::cerr << res[i].a << " " << res[i].b << "  " << res[i].similarity << std::endl; */
    /* } */
    std::stringstream key;
    key << "top-" << k << "-pairs";
    H5Easy::dump(file, key.str(), out_res, H5Easy::DumpMode::Overwrite);

    return 0;
}

