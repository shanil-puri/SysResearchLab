ax = gca;
B = csvread('/Users/Shanil/PCA_results/results_0.5/runtime/final_runT_v1.txt')
C = csvread('/Users/Shanil/PCA_results/results_0.75/runtime/final_runT_v1.txt') 
D = csvread('/Users/Shanil/PCA_results/results_0.9/runtime/final_runT_v1.txt') 
A = size(B)
% for k = 1:A(1)
t1 = plot(9, median(B),'red o');
hold on
t1 = plot(26, median(C),'red o');
hold on
t1 = plot(54, median(D),'red o');
hold on
    
E = csvread('/Users/Shanil/PCA_results/results_0.5/overhead/final_runT_v1.txt')
F = csvread('/Users/Shanil/PCA_results/results_0.75/overhead/final_runT_v1.txt') 
G = csvread('/Users/Shanil/PCA_results/results_0.9/overhead/final_runT_v1.txt') 
t2 = plot(9, median(B) + median(E),'green square');
hold on
t2 = plot(26, median(C) + median(F),'green square');
hold on
t2 = plot(54, median(D) + median(G),'green square');
hold on

H = csvread('/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/nn_data/K=600final_runT_v1.txt')
t1 = plot(3, median(H),'red o');

t2 = plot(3, 94141 + median(H), 'green square')


% end


% A = size(C)
% for k = 1:A(1)
%     t1 = plot(0., (C(k,1)/B(k,1)),'red o');
%     hold on
% end
% 
% 
% A = size(D)
% for j= 1:A(1)
%     t2 = plot(0.9, (D(j,1)/B(j,1)), 'green square');
%     hold on
% end

xLabels = [3 9 26 54]

[hleg1, hobj1] = legend([t1, t2],{'K-Means-HRu', 'K-Means-HRu + PCA Overhead'});
xlabel('PCA Dimentions Selected');
ylabel('Median Runtime')

set(gca, 'XTick', xLabels);
set(gca, 'XTickLabel', xLabels);
set(gca,'XScale','log') 


textobj = findobj(hobj1, 'type', 'text');
rect = [0.40, 0.70, 0.47, 0.10];
set(textobj, 'Interpreter', 'latex', 'fontsize', 14);
set(hleg1, 'Position', rect)


% legend('Caltec 101 Kmeans HR')
% hold on;

% legend('Caltec 101 Kmeans HR')
% legend('Blah')
% legend('Blah2')