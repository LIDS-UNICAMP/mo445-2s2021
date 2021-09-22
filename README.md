# MO445 Scripts

# Requirements

## FLIM requirements

Firstly, download the [FLIM](https://github.com/LIDS-UNICAMP/FLIM/tree/mo445) project at the *mo445* branch. For example:

```
  cd <git_dir>
  git clone git@github.com:LIDS-UNICAMP/FLIM.git
  cd FLIM
  git checkout mo445
  pip install -r requirements.txt
  pip install .
```


Once instaled,FLIM project provides an annotation tool for drawing markers at any specified image:

```
    annotation <image_path>
```

## This project requirements

To install this project requirements do:
```
    cd <tarefa_mo445>
    pip install -r requirements.txt
```


# How to use


1. Create the markers with the annotation tool, for example:

```
    annotation imgs_and_markers/9.png
```

2. Train the FLIM encoder for all images and markers in the provided folder, for example:

```
    python src/train_encoder.py -a arch-unet.json -i imgs_and_markers/ -o encoder.pt
```

2.5. The [extract_features.py](src/extract_features.py) extract features for an given pair of model-image. This will save all features in a specified folder, the features will be saved in .png and .mimg. **Note that the script will erase the output features folder to reduce disk space consumption and to not mix up features from different images. That said, use different output folders for different input images.**

```
    python src/extract_features.py -a arch-unet.json -i imgs_and_markers/9.png -o features/ -m encoder.pt

```


3. Execute and C program from [libmo445](libmo445.tar.bz2) that uses the markers created in the first step. This program should create an segmentation mask for each image in the '''<original_img>_label.png''' format.


4. Using the segmentation mask of the previous step, train a deep learning model with the [train_flim_unet.py](src/train_flim_unet.py)script, for example:  
```
    python src/train_flim_unet.py -a arch-unet.json -id imgs_and_markers/ -gd gts/ -ne 3
```

***Note: The [train_flim_unet.py](src/train_flim_unet.py) scripts expect that every _label.png image has an equivalent original image***

5. The script [eval_unet](src/eval_unet.py) evaluates the model trained in the previous step, showing the intersection over unit (IoU) from input set. The *-output* argument is optional and if provided will be used to save the all masks predicted by the flim_unet model.


```
    python src/eval_unet.py -a arch-unet.json -id imgs_and_markers/ -gd gts/ -o output/
```

