from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import os
from nnunet.dataset_conversion.utils import generate_dataset_json

if __name__ == '__main__':
    # this is the data folder from the oai-zib-mri.
    oai_data_dir = '/home/lincoln/Documents/Data/OAI-ZIB-MRI/nifti'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 510
    task_name = "oai_zib_mri"

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    case_ids = subdirs(oai_data_dir, join=False)
    for c in case_ids:
        if isfile(join(oai_data_dir, c, 'image.nii.gz')) and isfile(join(oai_data_dir,c,'mask.nii.gz')):
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
