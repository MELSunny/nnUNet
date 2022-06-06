#!/bin/bash
add-on="unittest2 nibabel scikit-image future matplotlib requests pandas tqdm scipy scikit-learn SimpleITK -c SimpleITK"
conda create -n pytorch pytorch torchvision torchaudio cudatoolkit=11.1 $add-on -c pytorch-lts -c nvidia