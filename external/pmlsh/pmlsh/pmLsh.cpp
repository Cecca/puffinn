#include "pmLsh.h"
#include "MyFunc.h"
#include <ctime>
#include <queue>


void pmLsh::improvedSearchWithKth(DataMetric& highData, DataMetric& highQueryData, DataMetric& lowQueryData, Real_Result& real_result)
{
	/*����*/
	cerr << "\nRUNNING PM-LSH ...\n\n";
	queryResult.clear();
	queryResult.resize(lowQueryData.size());

	progress_display pd(lowQueryData.size());
	for (int i = 0; i < lowQueryData.size(); ++i) {
		DATATYPE cost = 0;
		std::priority_queue<knnPair> cand;
		vector<bool> visitFlag(highData.size(), false);
		queryResult[i].reserve(highData.size());
		vector<pair<DATATYPE, int>> singleQueryResult;
		singleQueryResult.reserve(highData.size());
		DATATYPE range = Config::search_Radius;
		std::priority_queue<knnPair, std::vector<knnPair>, greater<knnPair>> buffer;

		myIndex->reset(highData.size());
		while (!myIndex->isCompleteIncrementalRangeSearch(range)) {
			myIndex->improvedRangeSearch1(highQueryData[i], highData, lowQueryData[i], range, buffer, cand, cost, visitFlag);
			range *= sqrt(Config::c_appro);
		}
		queryResult[i].resize(Config::KNN);
		int t = Config::KNN - 1;
		while (!cand.empty()) {
			queryResult[i][t--] = make_pair(cand.top().dist, cand.top().id);
			cand.pop();
		}
		++pd;
	}

	cerr << "\nFINISH QUERY!\n\n";
}


void pmLsh::constructIndex(DataMetric& lowData)
{
	cerr << "BUILDING INDEX...\n";
	Timer timer;
	vector<vector<DATATYPE>> pivot_vec = Pivot::Random_pivot(lowData, Config::pivotNum, Config::pivotRandomCount);
	myIndex->Set_Pivot(Config::pivotNum, Config::pivotNum, Config::pivotNum, pivot_vec);
	progress_display pd(lowData.size());
	for (int i = 0; i < lowData.size(); ++i) {
		myIndex->Insert(lowData[i], i);
		++pd;
	}
	cerr << "FINISH BUILDING WITH TIME: " << timer.elapsed() << " s.\n";
	//printf("The whole volume is: %f\n", myIndex.Traverse_Get_Volume());
}

