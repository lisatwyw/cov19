# Classification of disease severity using the COV19-CT-DS

## COV19-CT-DB

### [Data folders and file counts](https://docs.google.com/spreadsheets/d/1SoVfioBKj_ElEETEk7o7KK_vs6VEca8LLIYW0xXpSYY/)

Summary:

|Diagnosis| Subset| n | Count per severity class |
|:--|:--|:--|:--|
|noncovid| train  | | N/A |
| | val | |   N/A |
| | test | | N/A |
|covid| train | 460 | 132, 123, 166, 39 |
| | val | 101 | 31, 20, 45, 5 | 
| | test | 231 | N/A |


## Reproducing results

1. Installation
```
pip install -r requirements.txt 
```

2a. Run interactively in ipython or python 
```
$ ipython


In [1]: BS=128; IMGSET='all2'; MID='wideres101'; exec( open('icassp_sep.py').read() )
```

2b. Run non=interactively
```
$ python icassp_sep.py 1 0 16   

# MID=1
#
# FT=0 # FT=0 : Fine tune the last layer only
#      # FT=1 : Fine tune all layers
#
# BS=16 
```            

## Other resources

### Datasets
- https://www.eibir.org/covid-19-imaging-datasets/
- https://www.kaggle.com/code/utkarshsaxenadn/ct-scans-3d-data-3d-data-processing-3d-cnn#3D-Scans-Data-Loading

### Lung mask segmentation code
https://github.com/pzaffino/COVID19-intensity-labeling/blob/main/lungs_processing.py    
- uses Simple ITK
