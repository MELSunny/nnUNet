import os
import os.path as osp
import pickle
import torch
import matplotlib.pyplot as plt
settings=['/mnt/d355c581-74e0-4df1-a7a6-29367453230b/OAI-ZIB-MRI/nnUNet_trained_models_plain',
          '/mnt/d355c581-74e0-4df1-a7a6-29367453230b/OAI-ZIB-MRI/nnUNet_trained_models_no_crop_ADJUST_FILTERS_(32,192)']
name=[osp.basename(setting).split('nnUNet_trained_models_')[1] for setting in settings]
modes=['3d_fullres','3d_fullres']
task='Task510_oai_zib_mri'
engine='nnUNetTrainerV2__nnUNetPlansv2.1'
fold='0'

paths=[osp.join(setting,'nnUNet',mode,task,engine,'fold_'+fold,'model_final_checkpoint.model') for (setting,mode) in zip(settings, modes)]
pkls=[pickle.load(open(path+'.pkl', "rb")) for path in paths]
plot_stuff=[torch.load(path)['plot_stuff'] for path in paths]
plt.plot(plot_stuff[0][0], label = "loss "+name[0])
plt.plot(plot_stuff[1][0], label = "loss "+name[1])
x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-0.95,-0.75))
plt.legend()
plt.show()

plt.plot(plot_stuff[0][1], label = "eval_metric "+name[0])
plt.plot(plot_stuff[1][1], label = "eval_metric "+name[1])
x1,x2,_,_ = plt.axis()
plt.axis((x1,x2,-0.95,-0.75))
plt.legend()
plt.show()
pass
