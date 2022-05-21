from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data,preprocessing_output_dir
import os
from nnunet.dataset_conversion.utils import generate_dataset_json
import SimpleITK as sitk
import shutil
if __name__ == '__main__':
    # this is the data folder from the oai-zib-mri.
    oai_data_dir = '/home/lincoln/Documents/Data/OAI-ZIB-MRI/nifti'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 510
    task_name = "oai_zib_mri"

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    fold1_file=os.path.join(os.path.dirname(oai_data_dir),'2foldCrossValidation-List1.txt')
    fold2_file=os.path.join(os.path.dirname(oai_data_dir),'2foldCrossValidation-List2.txt')
    out_preprocessed = join(preprocessing_output_dir, foldername)
    maybe_mkdir_p(out_preprocessed)

    with open(fold1_file) as file:
        fold1_cases = [line.rstrip() for line in file]
    with open(fold2_file) as file:
        fold2_cases = [line.rstrip() for line in file]
    splits = [
        {'train': fold1_cases, 'val': fold2_cases},
        {'train': fold2_cases, 'val': fold1_cases}
    ]


    save_pickle(splits, join(out_preprocessed, "splits_final.pkl"))

    out_base = join(nnUNet_raw_data, foldername)
    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    case_ids = subdirs(oai_data_dir, join=False)

    for c in case_ids:
        image_path=join(oai_data_dir, c, 'image.nii.gz')
        label_path=join(oai_data_dir,c,'mask.nii.gz')
        # image=sitk.ReadImage(image_path)
        # label=sitk.ReadImage(label_path)
        # label.SetOrigin(image.GetOrigin())
        # label.SetSpacing(image.GetSpacing())
        # label.SetDirection(image.GetDirection())
        # sitk.WriteImage(label,label_path)

        if isfile(image_path) and isfile(label_path):
            os.symlink(join(oai_data_dir, c, 'image.nii.gz'), join(imagestr, c + '_0000.nii.gz'))
            os.symlink(join(oai_data_dir, c, 'mask.nii.gz'), join(labelstr, c + '.nii.gz'))

    generate_dataset_json(join(out_base, 'dataset.json'),
                          imagestr,
                          None,
                          ('MRI',),
                          {
                              0: 'background',
                              1: 'femoral bone',
                              2: 'femoral cartilage',
                              3: 'tibial bone',
                              4: 'tibial cartilage',
                          },
                          task_name,
                          license='see https://www.zib.de/impressum',
                          dataset_description='see https://pubdata.zib.de/',
                          dataset_reference='https://pubdata.zib.de/',
                          dataset_release='0')
