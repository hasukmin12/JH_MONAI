import os

import nibabel as nib

# 1. Proxy 불러오기


profxy = nib.load('/disk1/sukmin/dataset/Task302_KiPA/imagesTs/case_00050.nii.gz')

# proxy = nib.load('/disk1/sukmin/inf_rst/unet/case_00050/case_00050_seg.nii.gz')

# 2. Header 불러오기
header = proxy.header

# 3. 원하는 Header 불러오기 (내용이 문자열일 경우 숫자로 표현됨)
header_size = header['sizeof_hdr']

# 2. 전체 Image Array 불러오기
arr = proxy.get_fdata()

# 3. 원하는 Image Array 영역만 불러오기
sub_arr = proxy.dataobj[..., 0:5]


# print(arr.shape)
# arr = arr.transpose((1,2,0))

print(arr.shape)
print(arr.max())
print(arr.min())
# print(arr)