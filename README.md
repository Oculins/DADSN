# Morphology Prior Enhanced Teeth Segmentation for High Resolution Oral Scans
Journal of Biomedical and Health Informatics, 2025

## Requirements:
```
pip install -r requirements.txt
```
## Training

### Step 0: prepare dataset

All the IOS model (.obj, .stl) and the annotation file (.json) should be in one folder `DATASET`.

```
-Dataset/
 --mesh_001.stl
 --anno_001.json
 --mesh_002.stl
 --anno_002.json
```

### Step1: preprocess - IOPS

```
python step1_preprocess.py --data_root DATASET
```

### Step 2: train the model

```
python step2_train.py
```

## Test

```
python step3_test.py
```


## Inference on your private IOS data

move your private data to `/work/input`, and the segmentation results will be saved in `/work/output`

```
python predict.py
```

## Batch prediction

put all stl files under`/work/input`, run`predict_ply_output.py`£¬colored mesh results in ply format will be saved under `/work/output`

```
python predict_ply_output.py --ckp_path outputs\Teeth3DS\models\best_model.pth --work_dir work
```

## Citation

Please refer to below if this helps you.

```
@article{jiang2025morphology,
  title={Morphology Prior Enhanced Teeth Segmentation for High-Resolution Oral Scans},
  author={Jiang, Yuxian and Wang, Xiuying and Yang, Tao and Ji, Changkai and He, Lanshan and Liu, Yusheng and Wang, Wei and Liu, Min and Shi, Junyu and Guo, Huayan and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

