x = 0:1:10;
A = [x; x+10];
A
fileID = fopen('exp.txt','w');
fprintf(fileID,'%d\n',A);
fclose(fileID);