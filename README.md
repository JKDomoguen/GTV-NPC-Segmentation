# Automatic segmentation of NPC GTV from CT images

This repository proivdes source code for automatic segmentation of Gross Target Volume (GTV) of Nasopharynx Cancer (NPC) from CT images according to the following paper:

* Domoguen, J. K. L., Manuel, J. J. A., Cañal, J. P. A., & Naval Jr, P. C. (2022). Automatic segmentation of nasopharyngeal carcinoma on CT images using efficient UNet‐2.5 D ensemble with semi‐supervised pretext task pretraining. Frontiers in Oncology, 12.


This code is built on top of the work by:
* Haochen Mei, Wenhui Lei, Ran Gu, Shan Ye, Zhengwentai Sun, Shichuan Zhang and Guotai Wang. "Automatic Segmentation of Gross Target Volume of Nasopharynx Cancer using Ensemble of Multiscale Deep Neural Networks with Spatial Attention." NeuroComputing, accepted. 2020.


# Requirement
Hardware requirement
* GPU with atleast 11GB Video RAM

# Usage

```bash
pip install -r requirements.txt
```

## Dataset Conversion: DICOM to NiFTI
The raw dataset is in DICOM data format which is the target data format a fully deployed solution would need to return to in order to be visualized by standard visualization and planning tools used by radio-oncologists.

In contrast, the machine learning models expects an array data and the best data format is NiFTI. The function and tools that can help convert DICOM to NiFTI is found in *HeadNECK_GTV/Data_preprocessing/preprocess.py*

## Training
1. Set the value of `root_dir` as your `GTV_root` in `config/gtv_npc.cfg`. Add the path of `PyMIC` to `PYTHONPATH` environment variable (if you haven't done this). Then you can start trainning by running following command:
 
```bash
export PYTHONPATH=$PYTHONPATH:your_path_of_PyMIC
python ../../pymic/train_infer/train_infer.py train config/train_test.cfg
```

2. During training or after training, run the command `tensorboard --logdir model/2D5unet` and then you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage. 

## Testing and evaluation
1. After training, run the following command to obtain segmentation results of your testing images:

```bash
mkdir result
python ../../pymic/train_infer/train_infer.py test config/train_test.cfg
```
