import numpy as np
import os
import nibabel as nib
import shutil
join = os.path.join

path = '/disk1/sukmin/dataset/Task001_Multi_Organ'
train_path = join(path, 'labelsTr')
test_path = join(path, 'labelsTs')

train_list = next(os.walk(train_path))[2]
test_list = next(os.walk(test_path))[2]
train_list.sort()
test_list.sort()
print(train_list)
print(test_list)


r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0
total = 0

for case in train_list:
    print()
    print(case)
    case_path = join(train_path, case)
    seg = nib.load(case_path).get_fdata()
    x_axis = int(seg.shape[0])
    y_axis = int(seg.shape[1])
    z_axis = int(seg.shape[2])

    for x in range(0, x_axis):
        for y in range(0, y_axis):
            for z in range(0, z_axis):
                if seg[x][y][z] == 1:
                    r1 += 1
                elif seg[x][y][z] == 2:
                    r2 += 1
                elif seg[x][y][z] == 3:
                    r3 += 1
                elif seg[x][y][z] == 4:
                    r4 += 1
                elif seg[x][y][z] == 5:
                    r5 += 1

    total = r1 + r2 + r3 + r4 + r5

    print(r1/total)
    print(r2/total)
    print(r3/total)
    print(r4/total)
    print(r5/total)
    print()
    print((total / r1) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r2) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r3) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r4) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r5) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))





for case in test_list:
    print()
    print(case)
    case_path = join(test_path, case)
    seg = nib.load(case_path).get_fdata()
    x_axis = int(seg.shape[0])
    y_axis = int(seg.shape[1])
    z_axis = int(seg.shape[2])

    for x in range(0, x_axis):
        for y in range(0, y_axis):
            for z in range(0, z_axis):
                if seg[x][y][z] == 1:
                    r1 += 1
                elif seg[x][y][z] == 2:
                    r2 += 1
                elif seg[x][y][z] == 3:
                    r3 += 1
                elif seg[x][y][z] == 4:
                    r4 += 1
                elif seg[x][y][z] == 5:
                    r5 += 1

    total = r1 + r2 + r3 + r4 + r5

    print(r1/total)
    print(r2/total)
    print(r3/total)
    print(r4/total)
    print(r5/total)
    print()
    print((total / r1) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r2) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r3) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r4) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
    print((total / r5) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))





print()
print("final results")
print()
print(r1/total)
print(r2/total)
print(r3/total)
print(r4/total)
print(r5/total)
print()
print((total / r1) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
print((total / r2) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
print((total / r3) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
print((total / r4) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
print((total / r5) / (total / r1 + total / r2 + total / r3 + total / r4 + total / r5))
