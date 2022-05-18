
from evaluator import evaluate_folder


# 비교하려는 GT 폴더 결과
folder_with_gt = '/disk1/sukmin/dataset/Task302_KiPA/labelsTs'

# inference 된 결과
folder_with_pred = '/disk1/sukmin/eval_rst/kipa_unet'

labels = (0, 1, 2, 3, 4) # test 하고 싶은 라벨 입력

evaluate_folder(folder_with_gt, folder_with_pred, labels)


# 실행이 완료되면 folder_with_pred 경로에 summary.json이 생성됌

# scp를 활용해서 로컬에서 열어보면 된다.
# ex) scp -r -P 22 sukmin@10.10.10.14:/disk1/sukmin/eval_rst/kipa_unet /home/sukmin/Downloads
