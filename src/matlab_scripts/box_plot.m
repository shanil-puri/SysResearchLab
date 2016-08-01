files_pp = {
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/nostredam/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/test_gas/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/road_network/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/kellog/K=240final_runT_v1.txt',
            };
        
files_cust = {
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/nostredam/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/test_gas/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/road_network/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/kellog/K=240final_runT_v1.txt',
            };
        
files_raw = {
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/nostredam/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/test_gas/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/road_network/K=240final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/kellog/K=240final_runT_v1.txt',
            };
  plot_mat = []

cust_mat = []
kpp_mat = []
for i = 1:4
%     cust_mat = zeros(3, (size(cust_best)), 'double')
    files_pp{i}
    kpp_best = csvread(files_pp{i})
    cust_best = csvread(files_cust{i})
    kmr_best = csvread(files_raw{i})
    
    med_kmr = median(kmr_best)
    
    for j = 1:(size(cust_best))
         cust_mat(j,i) = med_kmr / cust_best(j,1)
         kpp_mat(j,i) = med_kmr / kpp_best(j,1)
    end
    
%     tmp_mat = cat(2, [med_cust med_kpp med_kmr])
%     plot_mat = vertcat(plot_mat, tmp_mat)
%     boxplot(cust_mat,'Whisker',1, 'DataLim', [-5,5]);
%       boxplot(cust_mat(1:(size(cust_best)),i),'DataLim', [0,5]);
%     boxplot(cust_mat)
    hold on
end
a =[]
a1 =[]
b =[]
b1 =[]
c =[]
c1 =[]
d = []
d1 = []

a = kpp_mat(1:20,1)
b = kpp_mat(1:43,2)
c = kpp_mat(1:35, 3)
d = kpp_mat(1:10,4)
% C(1:length(c), 1) = 2;
% D(1:length(d), 1) = 3;
a1 = cust_mat(1:20,1)
b1 = cust_mat(1:43,2)
c1 = cust_mat(1:35, 3)
d1 = cust_mat(1:10,4)
% 
% a = kpp_mat(1:20,1)
% b = kpp_mat(1:20,2)
% c = kpp_mat(1:20, 3)
% d = kpp_mat(1:10,4)

A = []
A1 = []
B = []
B1 = []
C = []
C1 = []

A = zeros(length(a),1)
A1(1:length(a1), 1) = 1;
B(1:length(b), 1) = 2;
B1(1:length(b1), 1) = 3;
C(1:length(c), 1) = 4;
C1(1:length(c1), 1) = 5;
D(1:length(d), 1) = 6;
D1(1:length(d1), 1) = 7;

z=[a1;a;b1;b;c1;c;d1;d]
% z=[a1;a;b1;b;c1;c]
g = [A; A1; B; B1; C; C1; D; D1];
% g = [A; A1; B; B1; C; C1];

% boxplot(z,g, 'Labels',{'Caltec 101', '', 'Tiny', '','Uk Bench', ''},'DataLim', [0,5])
boxplot(z,g, 'Labels',{'NotreDame','', 'Road Network', '','US Gas Sensor Data', '', 'Kegg Network', ''},'DataLim', [0,5])
% boxplot(cust_mat,'Labels',{'NotreDame', 'Road Network', 'Kegg Network'},'DataLim', [-5,5]);

title('Runtime performance comparison')
ylabel('Speedup (X times Median Runtime for K-means)')
xlabel('Data Sets')

ax = gca;
legend('History Reuse Kmeans', 'Kmeans++');
clear tmp_mat;
clear plot_mat;

hold on