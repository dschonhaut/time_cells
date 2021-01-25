clear;
for iChan = 1:96
    load(['times_CSC' num2str(iChan) '_removepli.mat']);
    clear rejected;
    save(['times_CSC' num2str(iChan) '_removepli.mat']);
    clear;
end