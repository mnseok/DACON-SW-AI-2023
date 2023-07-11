import torch
import torch.nn as nn
import cv2
import numpy as np

def s1_loss(outputs,labels):
    criterion = nn.BCELoss()
    bceLoss = criterion(outputs, labels)
    return bceLoss + BLoss(outputs, labels)

def BLoss(outputs, labels):
    return IoU(edge(outputs), edge(labels))

def IoU(ou_edge, la_edge):
    intersection = torch.logical_and(ou_edge, la_edge).sum()
    union = torch.logical_or(ou_edge, la_edge).sum()
    iou = intersection.float() / (union.float() + 1e-8)  # Adding a small epsilon to avoid division by zero
    return 1 - iou

def edge(input):
    img = input.detach().cpu().numpy()
    tmp = canny(img)
    output = torch.from_numpy(tmp)
    return output

def canny(img):
    tmp = cv2.erode(img, kernel=np.ones(shape=(3, 3), dtype=np.float32) * 2, iterations=3)
    tmp = cv2.Canny(tmp, 127, 255)
    tmp = cv2.dilate(tmp, kernel=np.ones(shape=(3, 3), dtype=np.float32) * 2, iterations=2)
    return tmp
