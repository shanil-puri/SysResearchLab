M = csvread('/Users/Shanil/comparison_nn_data/comparison_30_600.txt')
for i = 1:(size(M))
  for j= 1:(size(M,1))-2
     t0 = plot(i, M(i,j), 'b.');
     hold on
  end
end

% legend('Caltec 101 Kmeans HR')
% hold on;

B = csvread('/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/nn_data/K=600final_runT_v1.txt')
A = size(B)
for j= 1:A(2)
    t = plot(j, B(1,j), 'ro');
    hold on
end
% legend('Caltec 101 Kmeans HR')
% hold on;


C = csvread('/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/nn_data/K=600final_runT_v1.txt') 
A = size(C)
for k = 1:A(2)
    t1 = plot(k, C(1,k), 'g square');
    hold on
end

D = csvread('/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/nn_data/K=600final_runT_v1.txt') 
A = size(D)
for j= 1:A(1)
    t2 = plot(j, D(j,1), 'black square');
    hold on
end

ax = gca;

[hleg1, hobj1] = legend([t0, t, t1, t2],{'Non Selected Historical Data', 'Selected Data Set','K-Means++', 'K-Means(Random)'});
textobj = findobj(hobj1, 'type', 'text');
rect = [0.40, 0.75, 0.43, 0.15];
set(textobj, 'Interpreter', 'latex', 'fontsize', 14);
set(hleg1, 'Position', rect)
xlabel('Data Set Number');
ylabel('Run Time(micro seconds)')
% legend('Caltec 101 Kmeans HR')
% hold on;

% legend('Caltec 101 Kmeans HR')
% legend('Blah')
% legend('Blah2')