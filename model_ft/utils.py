import os
import yaml

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


def log_gradients_in_model(model, phase, step, train_logger):
    if phase == 'train':
        for tag, value in model.named_parameters():
            if value.grad is not None:
                train_logger.add_histogram(tag + f"/{phase}/grad", value.grad.cpu(), step)


def log_loss_metric(phase, loss, metric, step, step_name: str, train_logger, val_logger):
    if phase == 'train':
        logger = train_logger
        logger.add_scalar(f'Loss/{phase} per {step_name}', loss, step)
        logger.add_scalar(f'Metric/{phase} per {step_name}', metric, step)

    if phase == 'val':
        logger = val_logger
        logger.add_scalar(f'Loss/{phase} per {step_name}', loss, step)
        logger.add_scalar(f'Metric/{phase} per {step_name}', metric, step)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')


def make_yaml(pl_model, dataset, check_pointer_path):
    if not os.path.exists(check_pointer_path):
        os.mkdir(check_pointer_path)
    d_opt = pl_model.configure_optimizers()
    d_opt['optimizer'] = str(d_opt['optimizer']).split('\n')
    d_opt['lr_scheduler'] = str(d_opt['lr_scheduler']).split('\n')
    meta_data = {**d_opt, **dataset}
    meta_data_path = os.path.join(check_pointer_path, 'meta_data.yml')
    with open(meta_data_path, 'w') as outfile:
        yaml.dump(meta_data, outfile, default_flow_style=False)

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        overall_f1_score: the average f1-score.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """

    hist = _fast_hist(true, pred, num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    overall_f1 = overall_f1_score(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)

    return overall_acc, overall_f1, avg_jacc, avg_dice

def _fast_hist(true, pred, num_classes):
    true = true.long()
    true = true.squeeze(1)
    _, pred = torch.max(pred, 1)
    pred = pred.long()

    mask_ = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask_] + pred[mask_],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def overall_pixel_accuracy(hist, EPS=1e-10):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc

def overall_f1_score(hist, EPS=1e-10):
    """Computes the total pixel f1-score.
    Args:
        hist: confusion matrix.
    Returns:
        f1: the overall f1-score.
    """
    precision = hist[0][0] / (hist[0][0] + hist[0][1] + EPS)
    recall = hist[0][0] / (hist[0][0] + hist[1][0] + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    return f1

def jaccard_index(hist, EPS=1e-10):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard) #the mean of jaccard without NaNs
    return avg_jacc

def dice_coefficient(hist, EPS=1e-10):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

