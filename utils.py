import torch
import wandb
import numpy as np

def log_image_table(args, image, label, predict):
    mask_images = []
    image = image[0]
    label = torch.argmax(label, dim=0) if not label.size()[0] == 1 else label[0]
    predict = torch.argmax(predict, dim=0) if not predict.size()[0] == 1 else predict[0]

    frames = int(np.round(image.shape[-1]/3))
    for frame in range(frames,frames*2,2):
        mask_images.append(wandb.Image(image[...,frame].numpy(), masks={
            "ground_truth":{"mask_data":label[...,frame].numpy(),"class_labels":args.class_names},
            "predictions":{"mask_data":predict[...,frame].numpy(),"class_labels":args.class_names},
        }))
    return mask_images

def calc_mean_class(args, list_, metric_class='valid_dice'):
    all_mean = {}
    all_value = 0
    for class_ in range(args.channel_out-1):
        mean_ = 0
        for i in range(len(list_)):
            mean_ += list_[i][class_]
        all_mean.update({f'{metric_class}/{args.class_names[class_+1]}':mean_/len(list_)})
        all_value += mean_/len(list_)
    all_value /= (args.channel_out-1)
    return all_mean, all_value

def calc_confusion_metric(metric_name, confusion_matrix):
    tp, fp, tn, fn = confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3]
    p = tp + fn
    n = fp + tn
    # calculate metric
    metric = check_confusion_matrix_metric_name(metric_name)
    numerator: torch.Tensor
    denominator: Union[torch.Tensor, float]
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator

def check_confusion_matrix_metric_name(metric_name: str):
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")