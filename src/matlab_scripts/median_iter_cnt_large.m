files_pp = {
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=120final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=120final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=120final_runT_v1.txt'
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/nn_data_pp/K=600final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/tiny_nn_data_pp/K=600final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/pp_kmeans_raw/uk_bench_pp/K=600final_runT_v1.txt'
            };
files_cust = {
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=80final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=120final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=120final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=120final_runT_v1.txt'
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=240final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=360final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=480final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/nn_data/K=600final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/tiny_nn_data/K=600final_runT_v1.txt',
                '/Users/Shanil/formatted_results_iter/cust_kmeans/uk_bench/K=600final_runT_v1.txt'
            };
        
plot_mat = []

cust_mat = []
kpp_mat = []      
nostredame_mat_median = [];
test_gas_mat_median = [];
road_network_mat_median = [];
kellog_mat_median = []

nostredame_mat_kpp_median = [];
test_gas_mat_kpp_median = [];
road_network_mat_kpp_median = [];
kellog_mat_kpp_median = [];

nostredame_mat_krr_median = [];
test_gas_mat_krr_median = [];
road_network_mat_krr_median = [];
kellog_mat_krr_median = [];

j = 1;
k = 2;
l = 3;
m = 4;

for i = 1:6
    files_pp{j}
    kpp_best = csvread(files_pp{j})
    cust_best = csvread(files_cust{j})
%     kmr_best = csvread(files_raw{j})
    nostredame_mat_median(i) = median(cust_best);
    nostredame_mat_kpp_median(i) = median(kpp_best);
    nostredame_mat_krr_median(i) = 100;
    
    files_pp{k}
    kpp_best = csvread(files_pp{k})
    cust_best = csvread(files_cust{k})
%     kmr_best = csvread(files_raw{k})
    test_gas_mat_median(i) = median(cust_best);
    test_gas_mat_kpp_median(i) = median(kpp_best);
    test_gas_mat_krr_median(i) = 100;
    
    kpp_best = csvread(files_pp{l})
    cust_best = csvread(files_cust{l})
%     kmr_best = csvread(files_raw{l})
    road_network_mat_median(i) = median(cust_best);
    road_networkt_kpp_median(i) = median(kpp_best);
    road_network_krr_median(i) = 100;

    j = j + 3;
    k = k + 3;
    l = l + 3;
end


nostredame_mat_median
test_gas_mat_median
road_network_mat_median

nostredame_mat_kpp_median;
test_gas_mat_kpp_median;
road_networkt_kpp_median;

nostredame_mat_krr_median;
test_gas_mat_krr_median;
road_network_krr_median;