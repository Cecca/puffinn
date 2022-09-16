/// Computes and stores the top-1000 Jaccard similarities for each vector in the given HDF5 file

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "highfive/H5Easy.hpp"
#include "puffinn.hpp"
#include "puffinn/dataset.hpp"

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
        std::cerr << "USAGE: SampleCosine [--sample SIZE] <dataset>" << std::endl;
        return 1;
    }

    // load data
    std::cerr << "loading data from " << path << std::endl;
    H5Easy::File file(path, H5Easy::File::ReadWrite);
    std::vector<std::vector<float>> data = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
    size_t n = data.size();
    size_t dim = data[0].size();
    std::cerr << "loaded " << n
              << " vectors from hdf5 file, of dimension "
              << dim << std::endl;
    puffinn::Dataset<puffinn::UnitVectorFormat> dataset(dim);
    for (auto x : data) {
        dataset.insert(x);
    }
    std::cerr << "stored vectors in dataset" << std::endl;

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
        similarities[h] = puffinn::CosineSimilarity::compute_similarity(
            dataset[i],
            dataset[j],
            dataset.get_description()
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
