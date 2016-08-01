comp_files = {
                '/Users/Shanil/comparison_road_ntwk/comparison_0_120.txt',
                '/Users/Shanil/comparison_road_ntwk/comparison_1_120.txt',
                '/Users/Shanil/comparison_road_ntwk/comparison_2_120.txt',
                '/Users/Shanil/comparison_road_ntwk/comparison_3_120.txt',
                '/Users/Shanil/comparison_road_ntwk/comparison_4_120.txt',
             }

fin_comp = [];
row_1 = []
row_2 = []
row_3 = []
row_4 = []
row_5 = []
comp = []
for i = 1:43
    for j = 1:5
        comp = csvread(comp_files{j});
        if (j == 1)
            row_1 = comp(i,:)
        elseif (j==2)
            row_2 = comp(i,:)
        elseif (j==3)
            row_3 = comp(i,:)
        elseif (j==4)
            row_4 = comp(i,:)
        elseif (j==5)
            row_5 = comp(i,:)
        end
    end
    for l = 1:41
        tmp = [row_1(1,l); row_2(1,l); row_3(1,l); row_4(1,l); row_5(1,l)]
        fin_comp(i,l) = median(tmp)
    end
end

fileID = fopen('exp.txt','w');
fprintf(fileID,fin_comp);