import os
import os.path as osp
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
os.environ["CUDA_VISIBLE_DEVICES"] = ""
settings=['/mnt/192.168.164.65/mnt/D6CC4E50CC4E2AD7/Project/OAI_ZIB_MRI/nnUNet/nnUNet_trained_models_no_crop_ADJUST_FILTERS_16_384_4_12_ASPP',
          '/mnt/192.168.164.65/mnt/D6CC4E50CC4E2AD7/Project/OAI_ZIB_MRI/nnUNet/nnUNet_trained_models_no_crop_ADJUST_FILTERS_16_384_4_24_ASPP',
          '/mnt/192.168.164.65/mnt/D6CC4E50CC4E2AD7/Project/OAI_ZIB_MRI/nnUNet/nnUNet_trained_models_no_crop_ADJUST_FILTERS_16_384_8_24_ASPP']


def smooth(values,smoothingWeight=0.9):
    smoothed=[]
    last = values[0]
    for value in values:
        smoothed_val = last * smoothingWeight + (1 - smoothingWeight) * value  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val
    return smoothed
    #last = last * smoothingWeight + (1 - smoothingWeight) * nextVal

names=[]
for setting in  settings:
    if len(osp.basename(setting).split('nnUNet_trained_models_'))<2:
        names.append('current')
    else:
        name=osp.basename(setting).split('nnUNet_trained_models_')[1]
        name=name.replace('no_crop_','')
        name = name.replace('ADJUST_FILTERS_', 'feat_size')
        name = name.replace('16_384_', '')
        name = name.replace('_ASPP', '')
        names.append(name)
modes=['3d_fullres','3d_fullres','3d_fullres','3d_fullres','3d_fullres']
task='Task510_oai_zib_mri'
engine='nnUNetTrainerV2__nnUNetPlansv2.1'
fold='0'

paths=[]
for (setting,mode) in zip(settings, modes):
    if osp.exists(osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold,'model_final_checkpoint.model') ):
        paths.append(osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold,'model_final_checkpoint.model') )
    elif osp.exists(osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold,'model_latest.model') ):
        paths.append(osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold,'model_latest.model'))
    else:
        raise FileNotFoundError(osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold))

pkls=[pickle.load(open(path+'.pkl', "rb")) for path in paths]
plot_stuffs=[torch.load(path,map_location=torch.device("cpu"))['plot_stuff'] for path in paths]

for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(smooth(plot_stuff[0]), label = "loss "+name,linewidth=0.5)

x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-1,-0.8))
plt.legend()
plt.show()


for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(smooth(plot_stuff[1]), label = "eval loss "+name,linewidth=0.5)

x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-1,-0.8))
plt.legend()
plt.show()

for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(smooth(plot_stuff[1]), label = "eval loss "+name,linewidth=0.5)
for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(smooth(plot_stuff[0]), label = "loss "+name,linewidth=0.5)

x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-1,-0.8))
plt.legend()
plt.show()
pass