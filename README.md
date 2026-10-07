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

## Weights

download weights from

https://drive.google.com/file/d/1f98a3nUMBokiy1qWkkndXJ1cejKcmlUm/view?usp=sharing

and put them under `/outputs`

## Inference on your private IOS data

move your private data to `/work/input`, and the segmentation results will be saved in `/work/output`

```
python predict.py
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

