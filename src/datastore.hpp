/* 
    TODO: LIST
        1. Implement the function read data from the data files and load them into the dataDb Map.
        2. Correct the implementation for dimBinPoints. This is needs to be a map of map with bin_id and data points on a per dimension basis. IMPLEMENTED INCORRECTLY RIGHT NOW
        3. Implement generate histogram.
        4. Test functionality.
*/
#ifndef DATASTORE_HPP
#define DATASTORE_HPP

#include "iostream"
// #include "Python.h"
#include "string"
#include "vector"
#include "unordered_map"
#include "map"
#include <unordered_map>
#include <sys/types.h>
#include "utilities.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <dirent.h>
#include <errno.h>
#include "fstream"
#include <sstream>
#include "limits"
#include <ctime>
#include <cmath>
#include "algorithm"
#include "set"
//#include "ython.h"
#include <ctime>
#include <sys/time.h>
#include <iterator>

#include "t_test.hpp"
#include "kmeans.hpp"

// #include <sys/time.h>
using namespace std;

#define DEBUG           0
#define TOTAL_BINS      32
#define TOP_K_BINS      4
#define DIMENSIONALITY  3
#define PCA_MIN_DIM     1
#define BIN_RANGE       "lower"
#define BIN_HIST        "histogram"


// typedef struct _bin_data_type
// {
//     size_t index;

// }bin_data_type;

class dataDb
{
public:
    size_t store_id;
    
    cv::Mat dimBinPoints;

    cv::Mat data;

    cv::Mat centroids;

    cv::Mat labels;

    cv::Mat currParProj;

    cv::Mat currSelfProj;

    cv::PCA pca;

    int has_pca;

    double compactness;

    // These are histograms on a per-dimension per-bin basis. These will be used in T-Test calculations.
    std::map<size_t, std::map<size_t, float> > histograms;

    std::map<size_t, cv::Mat> PCA_bin_data_pts;

    std::map<size_t, size_t > PCA_bin_population;

    std::map<size_t, float> PCA_bin_lims;

    // Rank the other data-stores in relation to this data-store. 
    // Doing this we will need to calculate only the best match data stoe for current database and then calculate variance for top three relative highest scoring data sets.
    // The chances are that the other data-sets will rank in similar order to current data-store if this is the best data store for us.
    // STORE ORDER -> Increasing. Remember to traverse in reverse order.
    std::map<float, std::pair<size_t, float>, std::greater<float> > relativeRank;

    //############################################################ FUNCTION PROTOTYPES ############################################################
    static cv::Mat prepare_pca_data(cv::Mat data, std::map<size_t, std::map<size_t, float > > &histograms, std::map<size_t, std::vector<size_t> > rowIndex);
    static cv::Mat reduce_data_points(cv::Mat data, std::map<size_t, std::map<size_t, float > > histograms, std::map<size_t, std::vector<size_t> > rowIndex);
    static cv::PCA PCA_dim_reduction(cv::Mat data);
    static int getFilesInDirectory(std::vector<string> * files, const string dir);
    static int readDataStore(string filePath, dataDb *store, size_t);
    static cv::Mat prepare_pca_data(cv::Mat data, std::map<size_t, std::map<size_t, float > > &histograms);
    static void project_pca_data(dataDb *, dataDb *);
    static void rank_data_set(dataDb *, dataDb *);
    static cv::Mat compute_centroid(cv::Mat data, cv::Mat *, double *, int);
    // static cv::Mat rank_runt_time_data_set(dataDb *);
    static size_t rank_runt_time_data_set(dataDb *);
    static std::map<size_t, cv::Mat > PCA_bin_data(cv::Mat , std::map<size_t, size_t> &, std::map<size_t, float > &);
    static cv::Mat get_labels_from_centers(dataDb *, dataDb *);
    //############################################################ FUNCTION PROTOTYPES ############################################################ 
};


std::map<size_t, std::vector<float> > rangeMap;
std::vector<dataDb *> dataStore;
std::unordered_map<size_t, dataDb *> dataStoreHash;

int start_time_computation = 0;
long reduce_data_points_time = 0;
long prepare_pca_data_time = 0;
long rank_data_set_pca_time = 0;
long rank_data_set_t_test_time = 0;
long rank_runt_time_data_set_time = 0;
long pca_dim_reduction_time = 0;

cv::Mat dataDb::reduce_data_points(cv::Mat data, std::map<size_t, std::map<size_t, float > > histograms, std::map<size_t, std::vector<size_t> > rowIndex)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    cv::Mat binPoints;
    // Input row only once even though the dimentions may lie in top ranking bins in multiple dimensions.
    std::unordered_map<size_t, int> rowUsed;

    for (std::map<size_t, std::map<size_t, float> >::const_iterator iter = histograms.begin(); iter != histograms.end(); ++iter)
    {
        std::multimap<float, size_t> dst = flip_map(iter->second);
        int i = 0;
        
        for (std::multimap<float, size_t>::reverse_iterator it = dst.rbegin(); i < TOP_K_BINS && it != dst.rend(); ++it, ++i)
        {
            for(std::vector<size_t>::const_iterator rIt = rowIndex[it->second].begin(); rIt != rowIndex[it->second].end(); ++rIt)
            {
                if(rowUsed.find(*rIt) == rowUsed.end()){
                    binPoints.push_back(data.row(*rIt));
                    rowUsed[*rIt] = 1;
                }
            }
        }
    }

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    if(start_time_computation)
    {
        reduce_data_points_time += elapsed_utime;
    }
    //###################### BENCHMARKING CODE ##############################

    return binPoints;
}

cv::PCA dataDb::PCA_dim_reduction(cv::Mat data)
{
    //###################### BENCHMARKING CODE ##############################
    // long elapsed_seconds;
    // long elapsed_useconds;
    // long elapsed_utime;

    // struct timeval tempo, tempo1;
    // gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, DIMENSIONALITY);
    
     //###################### BENCHMARKING CODE ##############################
    // gettimeofday(&tempo1, NULL);

    // elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    // elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    // elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    // if(start_time_computation)
    // {
    //     pca_dim_reduction_time += elapsed_utime;
    // }
    //###################### BENCHMARKING CODE ##############################
    // cout << pca;
    return pca;
}

int dataDb::getFilesInDirectory(std::vector<string> * files, const string dir)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        // return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if(dirp->d_name[0] != '.')
            files->push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

// Once files have been read simply store into a map.
int dataDb::readDataStore(string filePath, dataDb *store, size_t ctr)
{
    // static size_t ctr = 0;
    ifstream file;

    file.open(filePath);
    if(!file){
        return -1;
    }
    
    std::vector<std::vector<float> > tmpVec;
    std::string numStream;
    
    cv::Mat tmpMat;

    while(std::getline(file, numStream))
    {
        std::size_t found = numStream.find_first_of(" ");       
        //numStream.erase(0, found);
        // Read the file line wise and generate a 2D cv::Mat. This will be used in PCA computation.
        std::istringstream buffer(numStream);
        std::vector<float> line((std::istream_iterator<float>(buffer)),
                             std::istream_iterator<float>());
        cv::Mat m = cv::Mat(line).t();
        tmpMat.push_back(m);
        
        // Calculate the range on a per-dimension basis while reading from the file.
        // This will be much faster re-reading the whole input and then calculating the range.
        int dim = 0;
        for (std::vector<float>::const_iterator iter = line.begin(); iter != line.end(); ++iter)
        {
            if(ctr == 0){
                rangeMap[dim].push_back(*iter);
                rangeMap[dim++].push_back(*iter);
                // cout << *iter;
            }
            else{
                rangeMap[dim][0] = (*iter < rangeMap[dim][0]) ? *iter : rangeMap[dim][0];
                rangeMap[dim][1] = (*iter > rangeMap[dim][1]) ? *iter : rangeMap[dim][1];
                dim++;
            }
        }
    }

    // cout << endl << endl << endl << endl << ctr <<endl<< tmpMat << endl << endl << endl;
    store->store_id = ctr++;
    store->data = tmpMat;
    return 0;
}


cv::Mat dataDb::prepare_pca_data(cv::Mat data, std::map<size_t, std::map<size_t, float > > &histograms)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    std::map<size_t, std::vector<size_t> > rowIndex;
    float range, binRange, lower_lim, upper_lim;
    for (int i = 0; i < data.cols; ++i)
    {
        range = rangeMap[i][1] - rangeMap[i][0];
        binRange = range / TOTAL_BINS;
        
        for (int j = 0; j < data.rows; ++j)
        {
            for (size_t k = 0; k < TOTAL_BINS; ++k)
            {
                lower_lim = ((float)k * binRange + rangeMap[i][0]);
                upper_lim = ((float)(k+1) * binRange + rangeMap[i][0]);

                if( lower_lim < data.at<float>(j,i) && upper_lim > data.at<float>(j,i) )
                {
                    rowIndex[k].push_back(j);
                    // Histogram for per-bin on a per-dimension basis.
                    if((histograms.find(i) != histograms.end()) && (histograms[i].find(k) != histograms[i].end())){
                        histograms[i][k]++;
                    }
                    else{
                        histograms[i][k] = 1;
                    }
                    break;
                }
            }
        }
    }

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    if(start_time_computation)
    {
        prepare_pca_data_time += elapsed_utime;
    }
    //###################### BENCHMARKING CODE ##############################
    // cout << "called oasbdcljads\n\n" << reduce_data_points(data, histograms, rowIndex);
    return reduce_data_points(data, histograms, rowIndex);
}


/* 
    1. Project child and parent data on parent PCA. -> Call project_pca_data
    2. Run T-test on reduced data sets. -> call SciPy T-Test library to implement this procedure.
    3. Used the variance to rank the child data-set wrt to the parent data-set.
*/
void dataDb::rank_data_set(dataDb *parent, dataDb *child)
{
    // //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds1;
    long elapsed_useconds1;
    long elapsed_utime1;

    struct timeval tempo2, tempo3;
    gettimeofday(&tempo2, NULL);
    
    //###################### BENCHMARKING CODE ##############################
    
    double _p_score = 0;
    int ctr = 0;

    std::vector<double> _sample1, _sample2;
    cv::Mat _parent_column, _child_column, _par_mat, _child_mat;

    // cout << "\n9a";
    std::map<size_t, size_t>::reverse_iterator _par_bin = parent->PCA_bin_population.rbegin();
    std::map<size_t, size_t>::reverse_iterator _chi_bin = child->PCA_bin_population.rbegin();
    while(ctr++ < 3)
    {
        _par_mat.push_back(parent->PCA_bin_data_pts[_par_bin->second]);
        _child_mat.push_back(child->PCA_bin_data_pts[_chi_bin->second]);
        _par_bin++; _chi_bin++;
    }

    for (int i = 0; i < _par_mat.cols; ++i)
    // for (int i = 0; i < 5; ++i)
    {
        _parent_column = parent->currSelfProj.col(i).rowRange(0, 5000).clone();
        _child_column = child->currSelfProj.col(i).rowRange(0, 5000).clone();
        _parent_column = _par_mat.col(i).clone();
        _child_column = _child_mat.col(i).clone();

        _parent_column.copyTo(_sample1);
        _child_column.copyTo(_sample2);

        _p_score += similarity_probability<double>(_sample1, _sample2);

        _sample1.clear();
        _sample2.clear();
    }

#if 0
    if(start_time_computation){
        cout << endl << parent->store_id << " : " << child->store_id << endl;
        cout << _p_score << endl;
    }
#endif
    
    parent->relativeRank[_p_score] = std::make_pair(child->store_id, _p_score);

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo3, NULL);

    elapsed_seconds1 = tempo3.tv_sec - tempo2.tv_sec;
    elapsed_useconds1 = tempo3.tv_usec - tempo2.tv_usec;
    elapsed_utime1 = (elapsed_seconds1) * 1000000 + elapsed_useconds1;
    // if(start_time_computation)
    // {
    //     cout << "Time taken : " << elapsed_utime1 << endl;
    // }
    //###################### BENCHMARKING CODE ##############################
}


std::map<size_t, cv::Mat > dataDb::PCA_bin_data(const cv::Mat pca_data, std::map<size_t, size_t> &bin_population, std::map<size_t, float > &bin_lims)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    
    //###################### BENCHMARKING CODE ##############################

    std::map<size_t, cv::Mat > _bin_data;
    std::map<size_t, float > _bin_lims;
    double _min, _max, _upper_lim, _data_pt;
    cv::Point min_loc, max_loc;
    // cout <<  endl << pca_data_col << endl; exit(1);
    cv::minMaxLoc((cv::InputArray)pca_data.col(0), &_min, &_max, &min_loc, &max_loc);

    // cout << _min << "\t" << _max << endl;
    double _range = _max - _min;
    double _bin_range = _range / TOTAL_BINS;

    size_t _bin;

    for (_bin = 0; _bin < TOTAL_BINS; ++_bin)
    {
        _upper_lim = ((float)(_bin+1) * _bin_range + _min);
        
        _bin_lims[_bin] = _upper_lim;
    }
    bin_lims = _bin_lims;
    for (int i = 0; i < pca_data.rows; ++i)
    {
        _data_pt = pca_data.at<float>(0, i);
        for (_bin = 0; _bin < TOTAL_BINS; ++_bin)
        {   
            if(_data_pt < _bin_lims[_bin]){
                _bin_data[_bin].push_back(pca_data.row(i));
                break;
            }
        }
    }
    for (std::map<size_t, cv::Mat >::iterator _it = _bin_data.begin(); _it != _bin_data.end(); ++_it)
    {
        bin_population[_it->second.rows] = _it->first;
    }

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    if(start_time_computation)
        cout << "Time Taken  bin data: " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
    return _bin_data;
}

size_t dataDb::rank_runt_time_data_set(dataDb *parent)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    long elapsed_seconds_gf;
    long elapsed_useconds_gf;
    long elapsed_utime_gf;

    struct timeval tempo, tempo1, tempo_gs, tempo_gf;
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo_gs, NULL);
    //###################### BENCHMARKING CODE ##############################

    dataDb *bestMatch_c1, *bestMatch_c2;
    float eigenDiff_c1, eigenDiff_c2, leastDiff_c1, leastDiff_c2;
    int ctr;
    double _index_diff, _tmp_index_diff;
    leastDiff_c1 = std::numeric_limits<float>::max();
    leastDiff_c2 = std::numeric_limits<float>::max();

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    parent->pca = dataDb::PCA_dim_reduction((parent)->data);
    parent->currSelfProj = parent->pca.project(parent->data);
    parent->PCA_bin_data_pts = dataDb::PCA_bin_data( parent->currSelfProj, parent->PCA_bin_population, parent->PCA_bin_lims );

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;

    cout << "\nParent Dataset PCA Projection and Histogram generation:\t" << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    int _row_size = parent->data.rows, _ch_data_pt_in_c1, _ch_data_pt_in_c2;
    size_t _parent_bin_index = 0, _child_bin_index_c1 = 0, _child_bin_index_c2 = 0;

    for (std::vector<dataDb *>::iterator childIter = dataStore.begin(); childIter != dataStore.end(); ++childIter)
    {
        eigenDiff_c1 = 0;
        eigenDiff_c2 = 0;
        ctr = 0;
        if ( (*childIter)->store_id != parent->store_id )
        {
            for (std::map<size_t, size_t>::reverse_iterator _bit = parent->PCA_bin_population.rbegin(); ctr < 1; ++_bit, ++ctr)
            {
                _parent_bin_index = _bit->second;
                _child_bin_index_c1 = _bit->second;
                // _child_bin_index = ((*childIter)->PCA_bin_population.rbegin())->second;
                _index_diff = std::numeric_limits<float>::max();
                // For the following store child in new address and then use top 2 for both above and below child address.
                for ( size_t _chi_bin = 0; _chi_bin < TOTAL_BINS; ++_chi_bin )
                {
                    _tmp_index_diff = fabs(pow(parent->PCA_bin_lims[_parent_bin_index] - (*childIter)->PCA_bin_lims[_chi_bin], 2));
                    if(_index_diff > _tmp_index_diff)
                        _child_bin_index_c2 = _chi_bin;
                }
                _ch_data_pt_in_c1 = 0;
                _ch_data_pt_in_c2 = 0;
                for (
                        int _data_pt_in = 0; 
                        _data_pt_in < parent->PCA_bin_data_pts[_bit->second].rows && _data_pt_in < parent->data.rows / 10;
                        ++_data_pt_in
                    )
                {
                    if( _ch_data_pt_in_c1 >= (int)(*childIter)->PCA_bin_data_pts[_child_bin_index_c1].rows - 1) {
                        while((int)(*childIter)->PCA_bin_data_pts[_child_bin_index_c1].rows == 0){
                            if(_child_bin_index_c1 >= TOTAL_BINS - 1)
                                _child_bin_index_c1 = 0;
                            else
                                _child_bin_index_c1++;
                        }
                        _ch_data_pt_in_c1 = 0;
                    }

                    if( _ch_data_pt_in_c2 >= (int)(*childIter)->PCA_bin_data_pts[_child_bin_index_c2].rows - 1) {
                        while((int)(*childIter)->PCA_bin_data_pts[_child_bin_index_c2].rows == 0){
                            if(_child_bin_index_c2 >= TOTAL_BINS - 1)
                                _child_bin_index_c2 = 0;
                            else
                                _child_bin_index_c2++;
                        }
                        _ch_data_pt_in_c2 = 0;
                    }                    

                    for(
                         int l = 0;
                         (l < parent->PCA_bin_data_pts[_parent_bin_index].cols && l < (*childIter)->PCA_bin_data_pts[_child_bin_index_c1].rows);
                         l++
                        )
                    {
                        eigenDiff_c1 += fabs(pow(parent->PCA_bin_data_pts[_parent_bin_index].at<float>(l, _data_pt_in), 2) - pow((*childIter)->PCA_bin_data_pts[_child_bin_index_c1].at<float>(l, _ch_data_pt_in_c1++),2));
                        eigenDiff_c2 += fabs(pow(parent->PCA_bin_data_pts[_parent_bin_index].at<float>(l, _data_pt_in), 2) - pow((*childIter)->PCA_bin_data_pts[_child_bin_index_c2].at<float>(l, _ch_data_pt_in_c2++),2));
                    }
                }
            }

            if(eigenDiff_c1 < leastDiff_c1)
            {
                leastDiff_c1 = eigenDiff_c1;
                bestMatch_c1 = (*childIter);
            }

            if(eigenDiff_c2 < leastDiff_c2)
            {
                leastDiff_c2 = eigenDiff_c2;
                bestMatch_c2 = (*childIter);
            }            
        }
    }

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nBest Match screen using histograms for comparison:\t" << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
    // exit(1);


    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    dataDb::rank_data_set(parent, dataStoreHash[bestMatch_c1->store_id]);
    dataDb::rank_data_set(parent, dataStoreHash[bestMatch_c2->store_id]);
    std::map<float, std::pair<size_t, float>, std::greater<float> >::iterator topK;
    ctr = 0;
    int ctr_1, ctr_2;
    std::unordered_map<size_t, int> _used_store_id;
    // gettimeofday(&tempo, NULL);
    ctr_1 = (leastDiff_c1 < leastDiff_c2) ? 2 : 1;
    ctr_2 = 3 - ctr_1;
    for (topK = bestMatch_c1->relativeRank.begin(); ctr < ctr_1 ;++topK)
    {
        size_t store_id = std::get<0>((*topK).second);
        if ( store_id != parent->store_id )
        {
            dataDb::rank_data_set(parent, dataStoreHash[store_id]);
            _used_store_id[store_id] = 1;
            ++ctr;
        }
    }
    ctr = 0;
    for (topK = bestMatch_c2->relativeRank.begin(); ctr < ctr_2 ;++topK)
    {
        size_t store_id = std::get<0>((*topK).second);
        if ( store_id != parent->store_id )
        {
            dataDb::rank_data_set(parent, dataStoreHash[store_id]);
            ++ctr;
        }
    }


    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nFinal selection using Welch's Test:\t" << elapsed_utime << endl;
    // if(start_time_computation)
    // {
    //     rank_runt_time_data_set_time += elapsed_utime;
    // }
    // size_t _final_selection = std::get<0>(parent->relativeRank.begin()->second);

    // dataDb *child = dataStoreHash.find(_final_selection)->second;
    // cout << "\nStore Id: " << child->store_id << endl;
    // cv::Mat labels = dataDb::get_labels_from_centers(parent, child);
    // return 1;

    // //###################### BENCHMARKING CODE ##############################
    // if(start_time_computation)
    //     cout << "Total overhead : " << elapsed_utime << endl;

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo_gf, NULL);

    elapsed_seconds_gf = tempo_gf.tv_sec - tempo_gs.tv_sec;
    elapsed_useconds_gf = tempo_gf.tv_usec - tempo_gs.tv_usec;
    elapsed_utime_gf = (elapsed_seconds_gf) * 1000000 + elapsed_useconds_gf;

    if(start_time_computation)
        cout << "Total overhead : " << elapsed_utime_gf << endl;
    //###################### BENCHMARKING CODE ##############################
    return std::get<0>(parent->relativeRank.begin()->second);
    // return labels;
}

cv::Mat dataDb::get_labels_from_centers(dataDb *parent, dataDb *child)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    
    //###################### BENCHMARKING CODE ##############################

    float _dist, _min_dist;
    int _cur_label;
    cv::Mat _labels;
    for (int i = 0; i < parent->currSelfProj.rows; ++i)
    {
        _min_dist = std::numeric_limits<float>::max();
        _cur_label = -1;
        for (int j = 0; j < child->centroids.rows; ++j)
        {
            _dist = 0;
            for (int k = 0; k < child->centroids.cols; ++k)
            {
                _dist += (float)pow( parent->currSelfProj.at<float>(i, k) - child->centroids.at<float>(j, k) , 2);
            }
            // _dist = pow( parent->currSelfProj.at<float>(i, 0) - child->centroids.at<float>(j, 0) , 2) +
            //         pow( parent->currSelfProj.at<float>(i, 1) - child->centroids.at<float>(j, 1) , 2) +
            //         pow( parent->currSelfProj.at<float>(i, 2) - child->centroids.at<float>(j, 2) , 2) +
            //         pow( parent->currSelfProj.at<float>(i, 3) - child->centroids.at<float>(j, 3) , 2) +
            //         pow( parent->currSelfProj.at<float>(i, 4) - child->centroids.at<float>(j, 4) , 2);

            if(_dist < _min_dist){
                _min_dist = _dist;
                _cur_label = j;
            }

        }
        _labels.push_back(_cur_label);
    }

    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    // if(start_time_computation)
    // cout << "Time Taken blah : " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################

    return _labels;
}

cv::Mat dataDb::compute_centroid(cv::Mat data, cv::Mat *labels, double *compactness, int cluster_count = 40)
{
    cv::Mat centers;
    cv::kmeans(data, cluster_count, *labels, cv::TermCriteria( cv::TermCriteria::EPS, 10000, .001), 3, cv::KMEANS_PP_CENTERS, centers);

    return centers;
}

std::vector<dataDb *> trainDataSet(const string dir, int cluster_count = 40)
{
    std::vector<string> v;

    dataDb::getFilesInDirectory(&v, dir);

    // Re-Initialise the test data store vector on every training run.
    dataStore.clear();
    dataStoreHash.clear();

    cout << "\n\nCluster Count : " << cluster_count << endl << endl;
    size_t _store_id = 0;
    for (int i = 0; i < (int)v.size(); ++i)
    {
        dataDb *tmpDataStore = new dataDb();
        dataDb::readDataStore(dir + "/" + v[i], tmpDataStore, _store_id++);
        dataStore.push_back(tmpDataStore);
    }

    int ctr = 0;
    for (std::vector<dataDb *>::iterator iter = dataStore.begin(); iter != dataStore.end(); ++iter)
    {
        dataStoreHash[(*iter)->store_id] = (*iter);
        (*iter)->dimBinPoints = dataDb::prepare_pca_data((*iter)->data, (*iter)->histograms);
        if((*iter)->data.cols > PCA_MIN_DIM)
        {
            // cout << "\n1\n";
            (*iter)->pca = dataDb::PCA_dim_reduction((*iter)->data);
            // cout << "\n2\n";
            (*iter)->has_pca = 1;
            // cout << "\n3\n";
            (*iter)->currSelfProj =  (*iter)->pca.project((*iter)->data);
            cout << "\n##########################################";
            cout << (*iter)->currSelfProj;
            cout << "\n##########################################";
            // cout << "\n4\n";
            (*iter)->PCA_bin_data_pts = dataDb::PCA_bin_data( (*iter)->currSelfProj, (*iter)->PCA_bin_population, (*iter)->PCA_bin_lims );
            // cout << "\n5\n";
            (*iter)->centroids = dataDb::compute_centroid((*iter)->currSelfProj, &(*iter)->labels, &(*iter)->compactness, cluster_count);
            // cout << "\n6\n";
            // exit(0);

#if DEBUG
    // cout << (*iter)->currSelfProj << endl<< endl;
    // // cv::Mat projMat = (*iter+1)->pca.project((*(iter+1))->data);
    // cout << endl<< endl << endl;
    // sleep(10);
    // // cout << (*iter)->pca.backProject(projMat) << "\n\n\n";
    // if(ctr > 1)
    //     exit(1);
    // ctr++;
#endif

        }
    }

    // exit(1);

    for (std::vector<dataDb *>::iterator parIter = dataStore.begin(); parIter != dataStore.end(); ++parIter)
    {
        for (std::vector<dataDb *>::iterator childIter = dataStore.begin(); childIter != dataStore.end(); ++childIter)
        {
            if(parIter != childIter)
            {
                // (*parIter)->currSelfProj = (*parIter)->pca.project((*parIter)->data);
                // cout << "\n7\n";
                (*childIter)->currParProj  = (*parIter)->pca.project((*childIter)->data.rowRange(0, (*parIter)->data.rows));
                // cout << "\n8\n";
                dataDb::rank_data_set(*parIter, *childIter);
                // cout << "\n9\n";
            }
        }
    }

#if DEBUG
    // std::map<float, std::pair<size_t, float>, std::greater<float> >::const_iterator rIter;
    // for (std::vector<dataDb *>::iterator parIter = dataStore.begin(); parIter != dataStore.end(); ++parIter)
    // {
    //     cout << "Par Store Id : " << (*parIter)->store_id << " : " << endl;
    //     for (rIter = (*parIter)->relativeRank.begin(); rIter != (*parIter)->relativeRank.end(); ++rIter)
    //     {
    //         cout << std::get<0>(rIter->second) << " : " << std::get<1>(rIter->second) << endl;
    //     }
    //     cout << endl << "#######################################" << endl;
    // }

    // exit(1);
#endif
    cout << "\n\n\n The end of training run is called\n\n\n";
    return dataStore;
}


#endif
