import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial import distance_matrix
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import torch
from torch import nn
import d2l

from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import json

from pycocotools.coco import COCO
import numpy as np
from torchvision.ops import nms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file) as f:
            data = json.load(f)

        if 'images' in data:
            self.images = data['images']
        if 'annotations' in data:  # Handling the presence of 'annotations'
            self.annotations = {image['id']: [] for image in self.images}
            for ann in data['annotations']:
                self.annotations[ann['image_id']].append(ann)
        else:
            self.annotations = {}
            print("Warning: 'annotations' key not found in JSON file.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        anns = self.annotations[img_info['id']]
        boxes = []
        for ann in anns:
            # Assuming 'bbox' is in the format [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.nelement() == 0:
            # 바운딩 박스가 없는 경우, 'area'는 빈 텐서가 됩니다.
            area = torch.empty((0,), dtype=torch.float32)
        else:
            # 바운딩 박스가 있는 경우, 'area'를 계산합니다.
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        #target["image_id"] = torch.tensor([img_info['id']])
        target["image_id"] = torch.tensor([1])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target




class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', self.get_blk(i))
            setattr(self, f'cls_{i}', self.cls_predictor(idx_to_in_channels[i],
                                                    self.num_anchors, self.num_classes))
            setattr(self, f'bbox_{i}', self.bbox_predictor(idx_to_in_channels[i],
                                                      self.num_anchors))


    # 모델 초기화 시 init_blocks 함수 호출
    # 예시: model = MyModel(); model.init_blocks()

    def multibox_prior(self, data, sizes, ratios):
        """Generate anchor boxes with different shapes centered on each pixel.

        Defined in :numref:`sec_anchor`"""
        in_height, in_width = data.shape[-2:]
        device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
        boxes_per_pixel = (num_sizes + num_ratios - 1)
        size_tensor = torch.tensor(sizes, device=device, dtype=torch.float32)
        ratio_tensor = torch.tensor(ratios, device=device, dtype=torch.float32)

        # Offsets are required to move the anchor to the center of a pixel. Since
        # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height  # Scaled steps in y axis
        steps_w = 1.0 / in_width  # Scaled steps in x axis

        # Generate all center points for the anchor boxes
        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # Generate `boxes_per_pixel` number of heights and widths that are later
        # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
        w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                       sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
            * in_height / in_width  # Handle rectangular inputs
        h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                       sizes[0] / torch.sqrt(ratio_tensor[1:])))
        # Divide by 2 to get half height and half width
        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
            in_height * in_width, 1) / 2

        # Each center point will have `boxes_per_pixel` number of anchor boxes, so
        # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                               dim=1).repeat_interleave(boxes_per_pixel, dim=0)
        output = out_grid + anchor_manipulations
        return output.unsqueeze(0)

    def base_net(self):
        blk = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            blk.append(self.down_sample_blk(num_filters[i], num_filters[i + 1]))
        return nn.Sequential(*blk)

    def down_sample_blk(self, in_channels, out_channels):
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))
        return nn.Sequential(*blk)

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            # 예측하기
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

    def cls_predictor(self, num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                         kernel_size=3, padding=1)

    def bbox_predictor(self, num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def get_blk(self, i):
        if i == 0:
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64, 128)
        elif i == 4:
            blk = nn.AdaptiveMaxPool2d((1,1))
        else:
            blk = self.down_sample_blk(128, 128)
        return blk

    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = self.multibox_prior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)

    def concat_preds(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)

    def flatten_pred(self, pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

class Trainer():
    def __init__(self, net, data_loader, device):
        self.net = net
        self.data_loader = data_loader
        self.device = device

    @staticmethod
    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls_loss = nn.CrossEntropyLoss(reduction='none')
        bbox_loss = nn.L1Loss(reduction='none')
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1, num_classes),
                       cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bbox_preds * bbox_masks,
                         bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox

    def show_tensor_image(self, image_tensor):
        # Check if the tensor is on a GPU and if so move it back to CPU
        if image_tensor.device != torch.device('cpu'):
            image_tensor = image_tensor.to('cpu')

        # Pytorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension, so we have to transpose the image
        image_tensor = image_tensor.transpose(0, 2).transpose(0, 1)

        # Make a new figure and set the figure size
        plt.figure(figsize=(10, 10))

        # Remove the axes
        plt.axis('off')

        # Use imshow to display the image
        plt.imshow(image_tensor)

        # Show the image
        plt.show()


    def train(self, num_epochs):
        min_loss = float('inf')
        self.net = self.net.to(self.device)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, weight_decay=5e-4)

        for epoch in range(num_epochs):
            self.net.train()
            running_loss = 0.0
            idx = 0
            for images, targets in self.data_loader:
                images = images.to(self.device)
                idx += len(images)

                stop = 1
                boxes = []
                label = []
                for target in targets:
                    if target['boxes'].dim() == 1:
                        stop = 0
                        print("버려")
                    boxes.append(target['boxes'].to(self.device))
                    label.append(target['image_id'].to(self.device))

                if (stop):
                    optimizer.zero_grad()

                    # Forward pass
                    anchors, cls_preds, bbox_preds = self.net(images)

                    # Compute bounding box loss (same note as above)
                    # Flatten all the boxes for the batch and the corresponding predictions
                    bbox_labels, bbox_masks, cls_labels = self.multibox_target(anchors, label, boxes)

                    l = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)

                    l.mean().backward()
                    optimizer.step()

                    running_loss += l.mean().item()

                    print(f'batch {len(self.data_loader.dataset)}/{idx}  | Loss: {l.mean().item():.4f}')

            epoch_loss = running_loss / len(self.data_loader)

            print('-------------------------------------------')
            print(f'Epoch {num_epochs}/{epoch + 1}, Loss: {epoch_loss:.4f}')
            print('-------------------------------------------')

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(self.net.state_dict(), 'best_model.pth')

    def box_iou(self, boxes1, boxes2):
        """Compute pairwise IoU across two lists of anchor or bounding boxes.

        Defined in :numref:`sec_anchor`"""
        if boxes2.dim() != 1:
            box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
            # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
            areas1 = box_area(boxes1)
            areas2 = box_area(boxes2)
            # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
            # boxes1, no. of boxes2, 2)
            inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
            inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
            inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
            # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
            inter_areas = inters[:, :, 0] * inters[:, :, 1]
            union_areas = areas1[:, None] + areas2 - inter_areas
            return inter_areas / union_areas
        else:
            return 0

    def assign_anchor_to_bbox(self, ground_truth, anchors, device, iou_threshold=0.5):
        """Assign closest ground-truth bounding boxes to anchor boxes.

        Defined in :numref:`sec_anchor`"""
        num_anchors, num_gt_boxes = anchors.shape[0], len(ground_truth)
        # Element x_ij in the i-th row and j-th column is the IoU of the anchor
        # box i and the ground-truth bounding box j
        jaccard = self.box_iou(anchors, ground_truth)
        # Initialize the tensor to hold the assigned ground-truth bounding box for
        # each anchor
        anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        # Assign ground-truth bounding boxes according to the threshold
        max_ious, indices = torch.max(jaccard, dim=1)
        anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
        box_j = indices[max_ious >= iou_threshold]
        anchors_bbox_map[anc_i] = box_j
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)  # Find the largest IoU
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_bbox_map[anc_idx] = box_idx
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return anchors_bbox_map


    def box_corner_to_center(self,boxes):
        """Convert from (upper_left, bottom_right) to (center, width, height)"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = torch.stack((cx, cy, w, h), axis=-1)
        return boxes

    def offset_boxes(self, anchors, assigned_bb, eps=1e-6):
        """Transform for anchor box offsets.

        Defined in :numref:`subsec_labeling-anchor-boxes`"""
        c_anc = self.box_corner_to_center(anchors)
        c_assigned_bb = self.box_corner_to_center(assigned_bb)
        offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], dim=1)
        return offset

    def multibox_target(self, anchors, labels, boxes):
        """Label anchor boxes using ground-truth bounding boxes.

        Defined in :numref:`subsec_labeling-anchor-boxes`"""
        batch_size, anchors = len(labels), anchors.squeeze(0)
        batch_offset, batch_mask, batch_class_labels = [], [], []
        device, num_anchors = anchors.device, anchors.shape[0]
        for i in range(batch_size):
            label = labels[i]
            anchors_bbox_map = self.assign_anchor_to_bbox(boxes[i], anchors, device)
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
                1, 4)
            # Initialize class labels and assigned bounding box coordinates with
            # zeros
            class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                       device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                      device=device)
            # Label classes of anchor boxes using their assigned ground-truth
            # bounding boxes. If an anchor box is not assigned any, we label its
            # class as background (the value remains zero)
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx].long()
            assigned_bb[indices_true] = label[bb_idx].float()
            # Offset transformation
            offset = self.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)
        return (bbox_offset, bbox_mask, class_labels)


def custom_collate_fn(batch):
    # Separate the images and the targets in the batch
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # You can stack the images because they have the same size
    images = default_collate(images)

    return images, targets


if __name__ == "__main__":
    # 이미지 전처리를 위한 변환 생성
    transform = transforms.Compose([
        transforms.Resize((360, 360)),  # 이미지 크기 조절
        transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환
    ])

    # 커스텀 이미지 데이터셋 로드
    dataset = CustomImageDataset(
        '/Users/apple/PycharmProjects/vit-ssd/ssd/idol/face.v9i.coco/train/_annotations.coco.json',
        '/Users/apple/PycharmProjects/vit-ssd/ssd/idol/face.v9i.coco/train',
        transform=transform
    )

    # DataLoader 생성
    train_iter = DataLoader(dataset, batch_size=32, shuffle=True , collate_fn=custom_collate_fn)

    # 학습을 위한 모델 초기화
    net = TinySSD(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trainer 인스턴스 생성 및 학습 시작
    trainer = Trainer(net, train_iter, device)
    trainer.train(num_epochs=20)

