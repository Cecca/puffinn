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
#include "pmlsh/pmLsh.h"
#include "pmlsh/Config.h"
#include "pmlsh/DataMetric.h"
#include "pmlsh/MyFunc.h"
#include "pmlsh/Real_Result.h"

#include "protocol.hpp"
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

Real_Result real_result;
DataMetric lowData;	
DataMetric lowQueryData;
DataMetric highData;
DataMetric highQueryData;

struct Pair
{
  size_t a;
  size_t b;
  float distance;
};

bool cmp_pairs(const Pair &a, const Pair &b)
{
  return a.distance < b.distance;
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

void load_data(std::vector<std::vector<float>>& data) {
	Timer timer;
	//highData.loadData(Config::highDataFilePath, Config::highDim);
	//highQueryData.loadData(Config::highQueryFilePath, Config::highDim);
    highData.setData(data);
    highQueryData.setData(data);
	cout << "FINISH LOADING WITH TIME: " << timer.elapsed() << " s.\n\n";

	cout << "HASHING...\n";
	timer.restart();
	E2LSH* e2lsh = new E2LSH;
	Config config;
	e2lsh->constructTable(config);
	lowData.lowerDimFromAnother(highData, e2lsh, config);
	lowQueryData.lowerDimFromAnother(highQueryData, e2lsh, config);
	cout << "FINISH HASHING WITH TIME: " << timer.elapsed() << " s.\n\n";
	Config::T = Config::T * highData.size();
	delete e2lsh;
}

int main(void) {

    // Read the dataset
    expect("data");
    expect("cosine");
    std::cerr << "[c++] receiving data" << std::endl;
    auto dataset = read_float_vectors_hdf5(true);
    std::cerr << "Loaded " << dataset.size() << " vectors of dimension " << dataset[0].size() << std::endl;
    send("ok");

    // Read parameters and build index
    std::string index_params_str = expect("index");
    std::cerr << "reading parameters from `" << index_params_str << "`" << std::endl;
    std::istringstream index_params_stream(index_params_str);

    Config::dataSetName = "test";
    Config::lowDim = 15;
    Config::highDim = dataset[0].size();
    Config::N = dataset.size();

	//pivot
	Config::pivotNum = 5;
	Config::pivotRandomCount = 5;

	//pm-lsh
	Config::c_appro = 1.5;
	Config::T = 0.1;
	Config::alpha1 = 0.01;
	Config::t = MyFunc::Ccal_thresh(Config::lowDim, Config::alpha1);
	Config::search_Radius = 3.0;


    size_t m = 16;
    double w = 4.0;
    size_t seed = 1234;
    while (true) {
        std::string key;
        index_params_stream >> key;
        std::cerr << "read key `" << key << "`" << std::endl;
        if (key == "") {
            break;
        }
        if (key == "m") {
            index_params_stream >> Config::lowDim;
        } else if (key == "w") {
            index_params_stream >> w;
        } else if (key == "seed") {
            index_params_stream >> seed;
        } else {
          // ignore other parameters
        }
    }

    if (Config::N > 9000000) {
		Config::MLeaf = 64;
		Config::M_NUM = 64;
	}

	if (Config::N > 50000000) {
		Config::MLeaf = 128;
		Config::M_NUM = 128;
	}

    load_data(dataset);
	pmLsh myPmlsh;
	myPmlsh.constructIndex(lowData);

    send("ok");

    while (true) {
      std::string next_workload = protocol_read();
      std::cerr << "received " << next_workload << std::endl;
      if (next_workload == "end_workloads") {
          break;
      }
      std::string workload_params_str = next_workload.substr(std::string("workload ").size());
      std::cerr << "NEW WORKLOAD ON INDEX " << workload_params_str << std::endl;

      // query params
      unsigned int k = 1;

      std::istringstream workload_params_stream(workload_params_str);
      while (true) {
          std::string key;
          workload_params_stream >> key;
          if (key == "") {
              break;
          }
          if (key == "k") {
              workload_params_stream >> Config::KNN;
          } else if (key == "T") { 
              workload_params_stream >> Config::T;
          } else if (key == "alpha1") { 
              workload_params_stream >> Config::alpha1;
	          Config::t = MyFunc::Ccal_thresh(Config::lowDim, Config::alpha1);
          } else if (key == "radius") { 
              workload_params_stream >> Config::search_Radius;
          } else {
              std::cout << "sppv1 err unknown parameter " << key << std::endl;
              throw std::invalid_argument("unknown parameter");
          }
      }

	  myPmlsh.improvedSearchWithKth(highData, highQueryData, lowQueryData, real_result);
      send("ok");
      auto res = myPmlsh.getResults();

      expect("result");
      std::cerr << "[c++] results size " << real_result.getSize() << std::endl; 
        for (auto v : res) {
            for (auto i : v) {
                std::cout << i.second << " ";
            }
            std::cout << std::endl;
        }
      send("end");
    }

    return 0;
}
