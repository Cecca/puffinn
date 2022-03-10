#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include "puffinn.hpp"

struct Dataset {
    std::vector<std::string> words;
    std::map<std::string, std::vector<float>> vectors;

    static Dataset read_glove(const std::string& filename);
};

void write_result(
    const std::string& filename, 
    const std::vector<std::vector<uint32_t>>& res
    ) {
    
    size_t n = res.size();
    size_t k = res[0].size();
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write((char*) &n, sizeof(size_t));
    fout.write((char*) &k, sizeof(size_t));
    for (auto& v: res) {
        fout.write((char*)&v[0], v.size() * sizeof(uint32_t));
    }
    fout.close();
}



const unsigned long long GB = 1024*1024*1024;
const unsigned long long MB = 1024*1024;

// Takes the following arguments: filename (num_neighbors) (recall) (space_usage in MB)
// The recall is a lower bound on the probability of finding each of the closest neighbors and is
// between 0 and 1.
int main(int argc, char* argv[]) {
    // Read parameters
    std::string filename;
    unsigned int k = 10;
    float recall = 0.8;
    std::string method = "BF";
    unsigned long long space_usage = 100*MB;
    switch (argc) {
        case 6: space_usage = static_cast<unsigned long long>(std::atof(argv[5])*MB); 
        case 5: method = std::string(argv[4]);
        case 4: recall = std::atof(argv[3]); 
        case 3: k = std::atoi(argv[2]);
        case 2: filename = argv[1];
                break;
        default:
            std::cerr << "Usage: " << argv[0]
                << " filename (number of neighbors) (recall) (BF|LSH) (space_usage in MB)" << std::endl;
            return -1;
    }

    // Read the dataset
    std::cerr << "Reading the dataset..." << std::endl;
    auto dataset = Dataset::read_glove(filename);
    if (dataset.words.size() == 0) {
        std::cerr << "Empty dataset" << std::endl;
        return -2;
    }
    auto dimensions = dataset.vectors[dataset.words[0]].size(); 

    // Construct the search index.
    // Here we use the cosine similarity measure with the default hash functions.
    // The index expects vectors with the same dimensionality as the first row of the dataset
    // and will use at most the specified amount of memory.
    puffinn::Index<puffinn::CosineSimilarity, puffinn::SimHash> index(
        dimensions,
        space_usage,
        puffinn::TensoredHashArgs<puffinn::SimHash>()
        // puffinn::IndependentHashArgs<puffinn::SimHash>()
    );
    // Insert each vector into the index.
    for (auto word : dataset.words) { index.insert(dataset.vectors[word]); }
    auto start_time = std::chrono::steady_clock::now();
    std::cerr << "Building the index. This can take a while..." << std::endl; 
    // Rebuild the index to include the inserted points
    index.rebuild();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end_time - start_time);
    auto throughput = ((float) dataset.words.size()) / elapsed.count();
    std::cerr << "Index built in " << elapsed.count() << " s " << throughput << " vecs/s" << std::endl;

    start_time = std::chrono::steady_clock::now();
    std::cerr << "Computing the join using " << method << ". This can take a while." << std::endl;    
    std::vector<std::vector<uint32_t>>  res;
    if (method == "BF") {
        res = index.bf_join(k);
    } else if (method == "LSH") {
        res = index.naive_lsh_join(k, recall);
    } else if (method == "LSHJoin") {
        res = index.lsh_join(k, recall);
    }
    end_time = std::chrono::steady_clock::now();
    elapsed = (end_time - start_time);
    throughput = ((float) dataset.words.size()) / elapsed.count();
    std::cerr << "Join computed in " << elapsed.count() << " s " << throughput << " queries/s" << std::endl;
    write_result(method, res);

}

// Read a vector collection in the format used by GloVe.
// Each line contains a word and a space-separated list of numbers. 
Dataset Dataset::read_glove(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::invalid_argument("File not found");
    }

    std::vector<std::string> words;
    std::map<std::string, std::vector<float>> vectors;
    while (!file.eof()) {
        std::string full_line;
        std::getline(file, full_line);
        std::istringstream line(full_line);
        
        std::string word;
        line >> word;

        std::vector<float> row;
        float val;
        while (line >> val) {
            row.push_back(val);
        }

        if (row.size() != 0) {
            words.push_back(word);
            vectors[word] = row;
        }
    }

    Dataset res;
    res.words = words;
    res.vectors = vectors;
    return res;
}
