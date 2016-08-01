M = csvread('/Users/Shanil/comparison_road_ntwk/comparison_1_120.txt')
for i = 1:(size(M))
  for j= 1:(size(M,1))-2
     t0 = plot(i, M(i,j), 'b.');
     hold on
  end
end

% legend('Caltec 101 Kmeans HR')
% hold on;

B = csvread('/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/road_network/K=120final_runT_v1.txt')
A = size(B)
for j= 1:A(2)
    t = plot(j, B(1,j), 'ro');
    hold on
end
% legend('Caltec 101 Kmeans HR')
% hold on;


C = csvread('/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/road_network/K=120final_runT_v1.txt') 
A = size(C)
for j= 1:A(2)
    t1 = plot(j, C(1,j), 'g square');
    hold on
end


D = csvread('/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/road_network/K=120final_runT_v1.txt') 
A = size(D)
for j= 1:A(2)
    t2 = plot(j, D(1,j), 'black square');
    hold on
end

legend([t0, t, t1, t2],{'Non Selected Historical Data', 'Selected Data Set','K-Means++', 'K-Means(Random)'});
xlabel('Data Set Number');
ylabel('Run Time(micro seconds)')

% legend('Caltec 101 Kmeans HR')
% hold on;

% legend('Caltec 101 Kmeans HR')
% legend('Blah')
% legend('Blah2')