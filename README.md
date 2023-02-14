# UCSF_FMRI_Connectone

 Winter 2023 UCSF_FMRI_Connectone project



There are some requirements of the Python environment. The python version is `3.8.5`.

```markdown
torch==1.9.0
pandas==1.3.2
numpy==1.19.5
```


The code of the project can be dirrectly run by the command below after adjusting the correct data path in the code `main.py.`

```markdown
python main.py
```



The data we use is in the folder `fMRI_connecomes`, while `fMRI_connecomes/ct` for control group and `fMRI_connecomes/ti` for tinnitus group. The original data of the control group and tinnitus group are in  `fMRI_connecomes/aroma30_2P_GSR_BN_control` and `fMRI_connecomes/aroma30_2P_GSR_BN_tinnitus` respectively, and we rename the data as label with index by  using data.py (after adjusting the correct data path) with the command below.

```markdown
python data.py
```

We store the two groups into `fMRI_connecomes/ct` and `fMRI_connecomes/ti` respectively. (To run the code, this step isn't needed since the data is already operated).


The summary of the project can be seen in the PDF file `Report.pdf`
