import os

all_file = os.listdir('/Users/mac/UCSF/fMRI_connecomes/control') # the folder of the original data
target_file = [x for x in all_file if x.endswith('.csv')]

print(len(target_file))
print(target_file)

for i in range(len(target_file)):
    srcFile = '/Users/mac/UCSF/fMRI_connecomes/control/'+target_file[i] # original data
    print(srcFile)
    dstFile = '/Users/mac/UCSF/fMRI_connecomes/ct/'+'ti'+str(i)+'.csv' # processed data
    os.rename(srcFile, dstFile)