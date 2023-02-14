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



The original data of the control group and tinnitus group are in  `fMRI_connecomes/aroma30_2P_GSR_BN_control` and `fMRI_connecomes/aroma30_2P_GSR_BN_tinnitus` respectively. We need to process the data as following in order to use it.

```
1. Create new folders "control" and "tinnitus".
2. Copy the data of the control group and the tinnitus group into the corresponding new folder.
3. Create new folders "ct" and "ti" for control group and tinnitus group respectively.
4. Run the code 'data.py' using the command below for control folder and tinnitus folder (remember to adjust the lines with annotations to maske sure the data paths are correct).
```

```markdown
python data.py
```

At last, we store the two groups into `fMRI_connecomes/ct` and `fMRI_connecomes/ti` respectively.


The summary of the project can be seen in the PDF file `Report.pdf`
