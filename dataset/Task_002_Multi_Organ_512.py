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
    This is the Multi_Organ from sukmin Ha
    """

    # base = '/nas3/sukmin/dataset'

    task_id = 2
    task_name = "Multi_Organ"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/nas3/sukmin/dataset'

    out_base = join(nnUNet_raw_data, foldername)
    # out_base = join(base, foldername)


    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")


    train_patient_names = []
    test_patient_names = []
    train_cases = next(os.walk(imagestr))[2]
    train_cases.sort()
    test_cases = next(os.walk(imagests))[2]
    test_cases.sort()

    for p in train_cases:
        train_patient_names.append(p)
    for p in test_cases:
        test_patient_names.append(p)


    json_dict = {}
    json_dict['name'] = "Multi_Oragen"
    json_dict['description'] = "Multi_Organ segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Multi_Oragen_for_Hutom"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Liver",
        "2": "Stomach",
        "3": "Pancreas",
        "4": "Gallbladder",
        "5": "Spleen"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1], "label": "./labelsTr/%s" % i.split("/")[-1]} for i in
                            train_patient_names]
    json_dict['test'] = ["./imagesTs/%s" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))