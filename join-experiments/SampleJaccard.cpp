/// Computes and stores the top-1000 Jaccard similarities for each vector in the given HDF5 file

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "highfive/H5Easy.hpp"

template<typename Iter>
float jaccard(Iter abegin, Iter aend, Iter bbegin, Iter bend) {
    size_t intersection = 0;
    size_t n_a = aend - abegin;
    size_t n_b = bend - bbegin;
    while (abegin != aend && bbegin != bend) {
        if (*abegin == *bbegin) {
            intersection++;
            abegin++;
            bbegin++;
        } else if (*abegin < *bbegin) {
            abegin++;
        } else {
            bbegin++;
        }
    }
    return ((float) intersection) / ((float) n_a + n_b - intersection);
}

int main(int argc, char ** argv) {
    std::string path;
    size_t samples = 1000000;

    if (argc == 2) {
        path = argv[1];
    } else if (argc == 4) {
        assert(argv[1] == "--sample");
        samples = atoi(argv[2]);
        path = argv[3];
    } else {
        std::cerr << "USAGE: SampleJaccard [--sample SIZE] <dataset>" << std::endl;
        return 1;
    }

    // load data
    std::cerr << "loading data from " << path << std::endl;
    H5Easy::File file(path, H5Easy::File::ReadWrite);
    std::vector<uint32_t> data = H5Easy::load<std::vector<uint32_t>>(file, "/train");
    std::vector<size_t> sizes = H5Easy::load<std::vector<size_t>>(file, "/size_train");
    size_t n = sizes.size();
    std::vector<size_t> offsets(n);
    size_t offset = 0;
    for (size_t i=0; i<n; i++) {
        offsets[i] = offset;
        offset += sizes[i];
    }

    std::mt19937_64 rng(1234);
    std::uniform_int_distribution<size_t> distrib(0, n-1);

    std::vector<std::pair<size_t, size_t>> indices;
    for (size_t h=0; h<samples; h++) {
        size_t i = distrib(rng);
        size_t j = distrib(rng);
        indices.push_back(std::make_pair(i, j));
    }

    std::vector<float> similarities(samples);

    size_t progress = 0;

    std::cerr << "computing similarities" << std::endl;
    // compute similarities
    #pragma omp parallel for schedule(dynamic)
    for (size_t h=0; h<indices.size(); h++) {
        size_t i = indices[h].first;
        size_t j = indices[h].second;
        similarities[h] = jaccard(
            data.begin() + offsets[i],
            data.begin() + offsets[i] + sizes[i],
            data.begin() + offsets[j],
            data.begin() + offsets[j] + sizes[j]
        );

        #pragma omp critical
        {
            if (++progress % 1000 == 0) {
                std::cerr << "completed " << progress << "/" << indices.size() << std::endl;
            }
        }
    }

    H5Easy::dump(file, "/sample-similarities", similarities);
}
