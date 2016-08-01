B = csvread('/Users/Shanil/compactness/k_rand_nn/final_runT_v1.txt')

C = csvread('/Users/Shanil/compactness/k_cust_nn/final_runT_v1.txt') 
A = size(C)
for k = 1:A(1)
    t1 = plot(k, (C(k,1)/B(k,1)),'red o');
    hold on
end

D = csvread('/Users/Shanil/compactness/k_pp_nn/final_runT_v1.txt') 
A = size(D)
for j= 1:A(1)
    t2 = plot(j, (D(j,1)/B(j,1)), 'green square');
    hold on
end

legend([t1, t2],{'K-Means-HRu', 'K-Means++'});
xlabel('Data Set Number');
ylabel('Normalized Compactness (Baseline K-Means(Rand))')
% legend('Caltec 101 Kmeans HR')
% hold on;

% legend('Caltec 101 Kmeans HR')
% legend('Blah')
% legend('Blah2')