import os
import shutil
import nibabel as nib
import numpy as np


path = '/disk1/sukmin/inf_rst/unet'
inf_list = next(os.walk(path))[1]
inf_list.sort()
print(inf_list)

aim_path = '/disk1/sukmin/eval_rst/kipa_unet'
if os.path.isdir(aim_path)==False:
    os.makedirs(aim_path)



for case in inf_list:
    print(case)
    input_path = os.path.join(path, case, case +'_seg.nii.gz')
    rename = case + '.nii.gz'
    output_path = os.path.join(aim_path, rename)

    seg = nib.load(input_path).get_fdata()
    x_axis = int(seg.shape[0])
    y_axis = int(seg.shape[1])
    z_axis = int(seg.shape[2])
    channel = int(seg.shape[3])
    new_seg = np.zeros((x_axis, y_axis, z_axis))
    
    for c in range(1, channel):
        for x in range(0, x_axis):
                for y in range(0, y_axis):
                    for z in range(0, z_axis):
                        if seg[x][y][z][c] != 0:
                            new_seg[x][y][z] = c

    xform = np.eye(4) * 2
    img_Nifti = nib.nifti1.Nifti1Image(new_seg, xform)
    nib.save(img_Nifti, output_path)
    print("saved format into : ", seg.shape, " -> ", new_seg.shape)
    print("saved nifti in : ", output_path)
    print("")