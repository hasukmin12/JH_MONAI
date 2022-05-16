#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import shutil
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == "__main__":
    """
    This is the KiPA from sukmin Ha
    """

    base = "/disk1/_has_KiPA"

    task_id = 302
    task_name = "KiPA"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/disk1/sukmin/dataset'

    out_base = join(nnUNet_raw_data, foldername)
    # out_base = join(base, foldername)


    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)

    train_patients = all_cases[:50]
    test_patients = all_cases[50:]

    for p in train_patients:
            curr = join(base, p)
            label_file = join(curr, "segmentation.nii.gz")
            image_file = join(curr, "imaging.nii.gz")
            shutil.copy(image_file, join(imagestr, p + ".nii.gz"))
            shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
            train_patient_names.append(p)

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + ".nii.gz"))
        test_patient_names.append(p)

    # 나중에 test inference를 위해 폴더는 만들어놓
    for p in test_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        shutil.copy(label_file, join(labelsts, p + ".nii.gz"))


    json_dict = {}
    json_dict['name'] = "KiPA"
    json_dict['description'] = "KiPA segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiPA data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Renal vein",
        "2": "Kidney",
        "3": "Renal artery",
        "4": "Tumor",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                            train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTs/%s.nii.gz" % i.split("/")[-1]} for i in
    #                    test_patient_names]
    save_json(json_dict, os.path.join(out_base, "dataset.json"))