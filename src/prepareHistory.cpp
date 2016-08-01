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

std::vector<dataDb *> read_test_datastore(const string dir)
{
    std::vector<string> v;
    std::vector<dataDb *> test_datastore;
    std::vector<cv::Mat> labels, centers;

    dataDb::getFilesInDirectory(&v, dir);
    for (int i = 0; i < v.size(); ++i)
    {
        dataDb *tmpDataStore = new dataDb();
        dataDb::readDataStore(dir + "/" + v[i], tmpDataStore);
        test_datastore.push_back(tmpDataStore);
    }
    return test_datastore;
}

void run_kmeans_no_lables(std::vector<dataDb *> datastore_test, int cluster_count = 40)
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

    int i = 0;
    for (std::vector<dataDb *>::iterator parIter = datastore_test.begin(); parIter != datastore_test.end(); ++parIter, ++i)
    {
        if(i < 3){
            cout << "K-means++ for data set no : " << i << endl;
            compactness = kmeans((*parIter)->data, cluster_count, label, cv::TermCriteria( cv::TermCriteria::EPS, 10000, .1), 1, cv::KMEANS_PP_CENTERS, center);
            cout << "compactness : " << compactness << endl;
            //cout << "The Labels Matrix" << label << endl;
        }
    }
    
    //###################### BENCHMARKING CODE ##############################
    gettimeofday(&tempo1, NULL);

    elapsed_seconds = tempo1.tv_sec - tempo.tv_sec;
    elapsed_useconds = tempo1.tv_usec - tempo.tv_usec;
    elapsed_utime = (elapsed_seconds) * 1000000 + elapsed_useconds;
    cout << "\nCPU Run Time in k means with no history reuse run : " << elapsed_utime << endl;
    //###################### BENCHMARKING CODE ##############################
}

void run_kmeans_custom_lables(std::vector<dataDb *> datastore_test, int cluster_count = 40)
{
    //###################### BENCHMARKING CODE ##############################
    long elapsed_seconds;
    long elapsed_useconds;
    long elapsed_utime;

    struct timeval tempo, tempo1;
    gettimeofday(&tempo, NULL);
    //###################### BENCHMARKING CODE ##############################
    std::vector<cv::Mat> labels, centers;
    int ctr = 0;

    for (std::vector<dataDb *>::iterator parIter = datastore_test.begin(); parIter != datastore_test.end(); ++parIter)
    {   
        if (ctr++ < 3)
        {
            cout << "History reuse K-means for data set no : " << ctr << endl;
            // (*parIter)->dimBinPoints = dataDb::prepare_pca_data((*parIter)->data, (*parIter)->histograms);

            if((*parIter)->data.cols > PCA_MIN_DIM)
            {
                (*parIter)->pca = dataDb::PCA_dim_reduction((*parIter)->data);
                (*parIter)->has_pca = 1;
            }
            
            size_t store_id = dataDb::rank_runt_time_data_set(*parIter);
            (*parIter)->compactness = kmeans((*parIter)->data, cluster_count, (dataStore[store_id])->labels, cv::TermCriteria( cv::TermCriteria::EPS, 10000, .1), 1, cv::KMEANS_USE_INITIAL_LABELS, (*parIter)->centroids);

            cout << "compactness : " << (*parIter)->compactness << endl;
            //cout << "the centroids input are : " << (*parIter)->centroids << endl;
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


void run_kmeans_no_lables_3d_small()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../3D_small_data";
    const string test_dir = "../3D_Small_Test_Set";

    datastore_test = read_test_datastore(test_dir);
    cout << "Running k-means++ algorithm: \n\n";
    run_kmeans_no_lables(datastore_test, 3);
}

void run_kmeans_custom_lables_3d_small()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../3D_small_data";
    const string test_dir = "../3D_Small_Test_Set";
    cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
    trainDataSet(dir, 3);
    cout << "Training of history data sets completed. \n";


    datastore_test = read_test_datastore(test_dir);

    cout << "Now running kmeans with custom labels : \n\n";
    run_kmeans_custom_lables(datastore_test, 3);
}

void run_kmeans_no_lables_3d_large()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../3D_large_data";
    const string test_dir = "../3D_Large_Test_Set";

    datastore_test = read_test_datastore(test_dir);
    cout << "Running k-means++ algorithm: \n\n";
    run_kmeans_no_lables(datastore_test, 80);
}
void run_kmeans_custom_lables_3d_large()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../3D_large_data";
    const string test_dir = "../3D_Large_Test_Set";

    cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
    trainDataSet(dir, 80);
    cout << "Training of history data sets completed. \n";

    datastore_test = read_test_datastore(test_dir);

    // cout << "Now running kmeans with custom labels with K-Index = 40: \n\n";
    // run_kmeans_custom_lables(datastore_test);

    cout << "Now running kmeans with custom labels : \n\n";
    start_time_computation = 1;
    run_kmeans_custom_lables(datastore_test, 80);

    cout << "\nReduce Data Points Time : " << reduce_data_points_time;
    cout << "\nPrepare PCA Data Time : " << prepare_pca_data_time;
    cout << "\nRank Dataset PCA Projection Time : " << rank_data_set_pca_time;
    cout << "\nRank data set T-Test Computation Time : " << rank_data_set_t_test_time;
    cout << "\nTotal computation over head : " << rank_runt_time_data_set_time << endl;
}

void run_kmeans_no_lables_10d_large()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../10D_large_data";
    const string test_dir = "../10D_Large_Test_Set";

    datastore_test = read_test_datastore(test_dir);
    cout << "Running k-means++ algorithm: \n\n";
    run_kmeans_no_lables(datastore_test, 80);
}
void run_kmeans_custom_lables_10d_large()
{
    std::vector<dataDb *> datastore_test;
    const string dir = "../10D_large_data";
    const string test_dir = "../10D_Large_Test_Set";

    cout << "Intiailizing training of history data sets: \t(This may take a while. Please be patient)\n";
    trainDataSet(dir, 80);
    cout << "Training of history data sets completed. \n";
    datastore_test = read_test_datastore(test_dir);

    // cout << "Now running kmeans with custom labels with K-Index = 40: \n\n";
    // run_kmeans_custom_lables(datastore_test);

    cout << "Now running kmeans with custom labels with K-Index = 80 : \n\n";
    start_time_computation = 1;
    run_kmeans_custom_lables(datastore_test, 80);

    cout << "\nReduce Data Points Time : " << reduce_data_points_time;
    cout << "\nPrepare PCA Data Time : " << prepare_pca_data_time;
    cout << "\nRank Dataset PCA Projection Time : " << rank_data_set_pca_time;
    cout << "\nRank data set T-Test Computation Time : " << rank_data_set_t_test_time;
    cout << "\nTotal computation over head : " << rank_runt_time_data_set_time + pca_dim_reduction_time << endl;
}


int main(int argc, char const *argv[])
{
    cout << "\n\n##########################################################################################################\n\n";
    cout << "Please use one of the following choices: \n";
    cout << "\tEnter 1 to run kmeans++ algorithm on untrained data (3D-Small): \n";
    cout << "\tEnter 2 to run kmeans algorithm with history reuse on data (3D-Small): \n";
    cout << "\tEnter 3 to run kmeans++ algorithm on untrained data (3D-Large): \n";
    cout << "\tEnter 4 to run kmeans algorithm with history reuse on data data (3D-Large): \n";
    cout << "\tEnter 5 to run kmeans++ algorithm on untrained data (10D-Large): \n";
    cout << "\tEnter 6 to run kmeans algorithm with history reuse on data(10D-Large): \n";
    cout << "\n\n##########################################################################################################\n\n";
    int choice = 0;
    cout << "Please enter your choice: \t";
    cin >>  choice;
    if(choice == 1)
        run_kmeans_no_lables_3d_small();
    else if(choice == 2)
        run_kmeans_custom_lables_3d_small();
    else if(choice == 3)
        run_kmeans_no_lables_3d_large();
    else if(choice == 4)
        run_kmeans_custom_lables_3d_large();
    else if(choice == 5)
        run_kmeans_no_lables_10d_large();
    else if(choice == 6)
        run_kmeans_custom_lables_10d_large();
    else{
        cout << "Wrong choice entered. Program exiting!!";
    }
    return 0;
}
    
