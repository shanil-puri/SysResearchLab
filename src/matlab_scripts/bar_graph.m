files_pp = {
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/nn_data/K=600final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/tiny_nn_data/K=600final_runT_v1.txt',
                '/Users/Shanil/fotmatted_results/analized_results/pp_kmeans_raw/uk_bench/K=600final_runT_v1.txt'
            };
        
files_cust = {
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/nn_data/K=600final_runT_v1.txt'
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/tiny_nn_data/K=600final_runT_v1.txt'
                '/Users/Shanil/fotmatted_results/analized_results/cust_kmeans/uk_bench/K=600final_runT_v1.txt'
            };
        
files_raw = {
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/nn_data/K=600final_runT_v1.txt'
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/tiny_nn_data/K=600final_runT_v1.txt'
                '/Users/Shanil/fotmatted_results/analized_results/kmeans_raw/uk_bench/K=600final_runT_v1.txt'
            };
  plot_mat = []
for i = 1:1
    kpp_best = csvread(files_pp{i})
    cust_best = csvread(files_cust{i})
    kmr_best = csvread(files_raw{i})

    med_cust = median(cust_best)
%     min_cust = min(cust_best)
%     max_cust = max(cust_best)

    med_kpp = median(kpp_best)
%     min_kpp = min(kpp_best)
%     max_kpp = max(kpp_best)

    med_kmr = median(kmr_best)
%     min_kmr = min(kmr_best)
%     max_kmr = max(kmr_best)
    
%     imp_perc_cust = 100 / med_cust;
%     imp_perc_kpp = 100 / med_kpp;

    % Use the below two to plot the error bar %
    % The "min" will give max speedup and the "max" represents best speedup. %
%     imp_min_cust = 100 / min_cust;
%     imp_min_kpp = 100 / min_kpp;

%     imp_max_cust = 100 / max_cust;
%     imp_max_kpp = 100 / max_kpp;
    
    tmp_mat = cat(2, [med_cust med_kpp med_kmr])
    plot_mat = vertcat(plot_mat, tmp_mat)
    %boxplot([cust_best, kpp_best, kmr_best],'Labels',{'mu = 5','mu = 6', 'mu = 6'},'Whisker',1)
%     boxplot(cust_best, Origin)
    hold on
end

% h = bar(plot_mat)
% 
% ax = gca;
% ax.XTickLabel = {'NN Data' 'Tiny' 'Uk Bench'};
% 
% legend('History Reuse Kmeans', 'Kmeans++');
% ax.YTick = [-1 -0.5 0 0.5 1];

% set(gca,'XTick', 1);
% set(gca,'XTickLabel','NN Data')
% hold on
% 
% set(gca,'XTick', 2);
% set(gca,'XTickLabel','Tiny')
% hold on
% 
% set(gca,'XTick', 3);
% set(gca,'XTickLabel','Uk Bench')

title('Runtime performance comparison')
ylabel('Median Runtime (microseconds)')
xlabel('Data Sets')

clear tmp_mat;
clear plot_mat;

% for i=1:length(h)
%     XDATA=get(get(h(i),'Children'),'XData');
%     YDATA=get(get(h(i),'Children'),'YData');
%     for j=1:size(XDATA,2)
%         x=XDATA(1,j)+(XDATA(3,j)-XDATA(1,j))/2;
%         y=YDATA(2,j)+ybuff;
%         t=[num2str(YDATA(2,j),3) ,'%'];
%         text(x,y,t,'Color','k','HorizontalAlignment','left','Rotation',90)
%     end
% end
% ylim([0 50])
hold on