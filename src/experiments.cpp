#include "iostream"
// #include "Python.h"
#include "string"
#include "vector"
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
#include "algorithm"
#include "set"
// #include "python.h"
#include <ctime>
#include <sys/time.h>
#include "datastore.hpp"


void run_kmeans_random_no_lables(std::vector<dataDb *> dataStore_test, int cluster_count = 40)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################

    std::vector<cv::Mat> labels, centers;
    cv::Mat label, center;
    double compactness;

    ofstream best_match;
    string best_match_file = "../results/krand_runtime_" + to_string(run_number) + "_" + to_string(cluster_count) + ".txt";
    best_match.open (best_match_file, std::ios::app);

    int i = 0;
    for (std::vector<dataDb *>::iterator parIter = dataStore_test.begin(); parIter != dataStore_test.end(); ++parIter, ++i)
    {
        gettimeofday(&tempo, NULL);
        cout << "K-means++ for data set no : " << (*parIter)->store_id << endl;
        compactness = kmeans((*parIter)->data, cluster_count, label, cv::TermCriteria( cv::TermCriteria::EPS, 10000, .01), 1, cv::KMEANS_RANDOM_CENTERS, center);
        cout << "compactness : " << compactness << endl;

        gettimeofday(&tempo1, NULL);
        elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
        elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
        elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
        cout << "CPU Run Time in k means with no history reuse run : " << elapsed_utime << endl << endl;
        best_match << elapsed_utime << endl;
    }
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nCPU Run Time in k means with no history reuse run : " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
}

void run_kmeans_no_lables(std::vector<dataDb *> dataStore_test, int cluster_count = 40)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################

    std::vector<cv::Mat> labels, centers;
    cv::Mat label, center;
    double compactness;

    ofstream best_match;
    string best_match_file = "../results/kpp_runtime_" + to_string(run_number) + "_" + to_string(cluster_count) + ".txt";
    best_match.open (best_match_file, std::ios::app);
    
    int i = 0;
    for (std::vector<dataDb *>::iterator parIter = dataStore_test.begin(); parIter != dataStore_test.end(); ++parIter, ++i)
    {
        gettimeofday(&tempo, NULL);
        cout << "K-means++ for data set no : " << (*parIter)->store_id << endl;
        compactness = kmeans((*parIter)->data, cluster_count, label, cv::TermCriteria( cv::TermCriteria::EPS, 10000, .01), 1, cv::KMEANS_PP_CENTERS, center);
        cout << "compactness : " << compactness << endl;

        gettimeofday(&tempo1, NULL);
        elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
        elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
        elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
        cout << "CPU Run Time in k means with no history reuse run : " << elapsed_utime << endl << endl;
        best_match << elapsed_utime << endl;
    }
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nCPU Run Time in k means with no history reuse run : " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
}

void run_kmeans_custom_lables(std::vector<dataDb *> dataStore_test, int cluster_count = 40)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1, tempo2, tempo3;
    // gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    std::vector<cv::Mat> centers;
    cv::Mat labels;

    ofstream best_match;
    string best_match_file = "../results/kcust_runtime_" + to_string(run_number) + "_" + to_string(cluster_count) + ".txt";
    best_match.open (best_match_file, std::ios::app);
    
    ofstream comparison;
    string comparison_file = "../results_0.5/kcust_comparison_" + to_string(run_number) + "_" + to_string(cluster_count) + ".txt";
    comparison.open(comparison_file, std::ios::app);

    for (std::vector<dataDb *>::iterator parIter = dataStore_test.begin(); parIter != dataStore_test.end(); ++parIter)
    {   

        cout << "\nHistory reuse K-means for data set no : " << (*parIter)->store_id << endl;
        (*parIter)->relativeRank.clear();
        size_t store_id = dataDb::rank_runt_time_data_set(*parIter);
        // cout << "\nStore Id : " << store_id << endl;
        gettimeofday(&tempo, NULL);
        (*parIter)->compactness = kmeans((*parIter)->data, cluster_count, dataStore[store_id]->labels, cv::TermCriteria( cv::TermCriteria::EPS, 10000, 0.01), 1, cv::KMEANS_USE_INITIAL_LABELS, (*parIter)->centroids);
        // cout << "compactness : " << (*parIter)->compactness << endl;
        gettimeofday(&tempo1, NULL);

        elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
        elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
        elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
        cout << "CPU Run Time in k means with history reuse run : " << elapsed_utime << endl;
        best_match << elapsed_utime << endl;

        cout << "\n\n#######################################################################################\n\n";
        cout << "################ \t\tRank check :\t\t ###################";
        cout << "\n\n#######################################################################################\n\n";
        for (int i = 0; i < (int)dataStore.size(); ++i)
        {
            gettimeofday(&tempo, NULL);
            (*parIter)->compactness = kmeans((*parIter)->data, cluster_count, dataStore[i]->labels, cv::TermCriteria( cv::TermCriteria::EPS, 10000, 0.01), 1, cv::KMEANS_USE_INITIAL_LABELS, (*parIter)->centroids);
            cout << "compactness : " << (*parIter)->compactness << endl;
            gettimeofday(&tempo1, NULL);
            elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
            elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
            elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
            cout << "CPU Run Time in k means with history reuse run with store : " << dataStore[i]->store_id << " : par id : " << (*parIter)->store_id <<" : " << elapsed_utime << endl;
            comparison << elapsed_utime << endl;
        }
    }
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nCPU Run Time in k means with history reuse run : " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
}

void test_road_newtwork_data()
{
    std::vector<dataDb *> dataStore_test;
    const string dir = "../3D_large_data";
    const string test_dir = "../3D_large_data";
    int cluster_sizes[] = {40, 80, 120, 240};
    
    for (int k = 0; k < 4; ++k)
    {
        for (int j = 0; i < 10; ++i){
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            trainDataSet(dir, cluster_sizes[k]);
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)43; ++i)
            {
                cout << "\n\nthe value of i is : " << i << endl << endl;
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, 240);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k]);
            }
        }
    }
}

void test_kellog_data()
{
    std::vector<dataDb *> dataStore_test;
    
    const string dir = "../10D_large_data";
    const string test_dir = "../10D_large_data";

    int cluster_sizes[] = {40, 80, 120, 240};
    
    for (int k = 0; k < 4; ++k)
    {
        for (int j = 10; i < 20; ++i){
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            trainDataSet(dir, cluster_sizes[k]);
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)19; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, 240);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k]);
            }
        }
    }
}


void test_gas_source_data()
{
    std::vector<dataDb *> dataStore_test;

    const string dir = "../dataset_two_gas_sources_/dataset_twosources_downsampled";
    const string test_dir = "../dataset_two_gas_sources_/dataset_twosources_downsampled";
    int cluster_sizes[] = {40, 80, 120, 240};
    
    for (int k = 0; k < 4; ++k)
    {
        for (int j = 20; i < 30; ++i){
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            trainDataSet(dir, cluster_sizes[k]);
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)35; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k]);
            }
        }
    }
}

void test_nn_data()
{
    std::vector<dataDb *> dataStore_test;

    const string dir = "../full_datasets/1_mil_nndata";
    const string test_dir = "../full_datasets/1_mil_nndata";
    int cluster_sizes[] = {80, 120, 240, 360, 480, 600};
    
    for (int k = 0; k < 6; ++k)
    {
        for (int j = 30; i < 40; ++i){
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            trainDataSet(dir, cluster_sizes[k]);
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)35; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k]);
                
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k]);
            }
        }
    }
}


void test_tiny_nn_data()
{
    std::vector<dataDb *> dataStore_test;

    const string dir = "../full_datasets/tiny2.nndata_shuffle";
    const string test_dir = "../full_datasets/tiny2.nndata_shuffle";
    int cluster_sizes[] = {80, 120, 240, 360, 480, 600};

    for (int k = 0; k < 6; ++k)
    {
        trainDataSet(dir, cluster_sizes[k]);
        for (int j = 40; j < 50; ++j)
        {
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)20; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, cluster_sizes[k], j);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k], j);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k], j);
            }
        }
    }
}

void test_uk_bech_nn_data()
{
    std::vector<dataDb *> dataStore_test;

    const string dir = "../full_datasets/uk_bench_nn_data";
    const string test_dir = "../full_datasets/uk_bench_nn_data";
    int cluster_sizes[] = {80, 120, 240, 360, 480, 600};

    for (int k = 0; k < 6; ++k)
    {
        trainDataSet(dir, cluster_sizes[k]);
        for (int j = 50; j < 60; ++j)
        {
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)20; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, cluster_sizes[k], j);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k], j);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k], j);
            }
        }
    }
}

void test_notredame_nn_data()
{
    std::vector<dataDb *> dataStore_test;

    const string dir = "../full_datasets/notredame";
    const string test_dir = "../full_datasets/notredame";
    int cluster_sizes[] = {40, 80, 120, 240};

    for (int k = 0; k < 4; ++k)
    {
        trainDataSet(dir, cluster_sizes[k]);
        for (int j = 60; j < 70; ++j)
        {
            cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
            cout << "Training of history data sets completed. \n";
            for (size_t i = (size_t)0; i < (size_t)20; ++i)
            {
                start_time_computation = 0;

                dataStore_test.clear();
                dataStore_test.push_back(dataStoreHash[i]);
                trainDataSet(dir, cluster_sizes[k]);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with Random Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_random_no_lables(dataStore_test, cluster_sizes[k], j);

                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with kmeans ++ Intiailizing :";
                cout << "\n############################################################################################\n\n";
                run_kmeans_no_lables(dataStore_test, cluster_sizes[k], j);
                cout << "\n\n############################################################################################\n";
                cout << "now running kmeans with custom labels :";
                cout << "\n############################################################################################\n\n";
                start_time_computation = 1;
                sleep(2);
                run_kmeans_custom_lables(dataStore_test, cluster_sizes[k], j);
            }
        }
    }
}

int main(int argc, char const *argv[])
{

    test_kellog_data();
    test_road_newtwork_data();
    test_gas_source_data();
    test_nn_data();
    test_tiny_nn_data();
    test_uk_bech_nn_data();
    test_notredame_nn_data();

    return 0;
}
    
