clear;
for iChan = 2:80
    load(['times_CSC' num2str(iChan) '.mat']);
    clear rejected;
    save(['times_CSC' num2str(iChan) '.mat']);
    clear;
end