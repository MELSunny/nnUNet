import os
import os.path as osp
import pickle
import torch
import matplotlib.pyplot as plt
settings=['/mnt/d355c581-74e0-4df1-a7a6-29367453230b/OAI-ZIB-MRI/nnUNet_trained_models',
          '/mnt/d355c581-74e0-4df1-a7a6-29367453230b/OAI-ZIB-MRI/nnUNet_trained_models_no_crop_ADJUST_FILTERS_16_384']
names=[]
for setting in  settings:
    if len(osp.basename(setting).split('nnUNet_trained_models_'))<2:
        names.append('current')
    else:
        names.append(osp.basename(setting).split('nnUNet_trained_models_')[1])
modes=['3d_fullres','3d_fullres','3d_fullres']
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
plot_stuffs=[torch.load(path)['plot_stuff'] for path in paths]

for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(plot_stuff[0], label = "loss "+name,linewidth=0.5)

x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-0.95,-0.75))
plt.legend()
plt.show()
for (plot_stuff,name) in zip(plot_stuffs,names):
    plt.plot(plot_stuff[1], label = "eval_metric "+name,linewidth=0.5)

x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-0.95,-0.75))
plt.legend()
plt.show()
pass
