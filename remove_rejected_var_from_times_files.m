clear;
for iChan = 5:5
    load(['times_CSC' num2str(iChan) '_removepli.mat']);
    clear rejected;
    save(['times_CSC' num2str(iChan) '_removepli.mat']);
    clear;
end