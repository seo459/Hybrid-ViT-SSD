import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels.

    Defined in :numref:`sec_object-detection-dataset`"""
    data_dir = '/Users/apple/PycharmProjects/vit-ssd/ssd/banana_apple/banana-detection'
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                 [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', self.get_blk(i))
            setattr(self, f'cls_{i}', self.cls_predictor(idx_to_in_channels[i],
                                                    self.num_anchors, self.num_classes))
            setattr(self, f'bbox_{i}', self.bbox_predictor(idx_to_in_channels[i],
                                                      self.num_anchors))

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

    def bbox_predictor(self, num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def cls_predictor(self, num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                         kernel_size=3, padding=1)

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

    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = self.multibox_prior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)

    def get_blk(self, i):
        if i == 0:
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64, 128)
        elif i == 4:
            blk = nn.AdaptiveMaxPool2d((1, 1))
        else:
            blk = self.down_sample_blk(128, 128)
        return blk

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

    def flatten_pred(self, pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    def concat_preds(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)




# 이미지 변환을 위한 transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# 데이터셋 로드 함수 정의
def load_data_bananas(batch_size_t,batch_size_v):
    """Load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size_t, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size_v)
    return train_iter, val_iter

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes.

    Defined in :numref:`sec_anchor`"""
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

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.

    Defined in :numref:`sec_anchor`"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
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

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset

def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
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
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right).

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset

def display(img_tensor, output, idx_img, threshold=0.9):
    # Select the first image from the batch
    img_tensor = img_tensor[0]

    # Convert the tensor image to a PIL image
    to_pil = torchvision.transforms.ToPILImage()
    img = to_pil(img_tensor.cpu())

    # Set the figure size
    fig, ax = plt.subplots(figsize=(5, 5))

    # Display the image
    ax.imshow(img)

    # Loop over each output row
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue

        # Get image height and width
        h, w = img.size

        # Create bounding box
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]

        # Show bounding boxes
        for box in bbox:
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            ax.text(box[0], box[1], '%.2f' % score, color='white')

    # Save the image
    plt.savefig(f'/Users/apple/PycharmProjects/vit-ssd/ssd/banana_apple/val_images/{idx_img}.jpg')
    plt.close(fig)


idx_img = 0

num_epochs =  20
device, net = 'cpu', TinySSD(num_classes=2)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 데이터셋 로드
train_iter, val_iter = load_data_bananas(batch_size_t=32, batch_size_v=1)

net = net.to(device)

# NMS안들어있음
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    net.train()
    train_loss_sum = 0
    idx = 0
    for features, target in train_iter:
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        train_loss_sum += l.mean().item()
        idx += 1
        print(f'epoch : {num_epochs}/{epoch}  |  batch :  {len(train_iter.dataset)//32+1}/{idx}  |  Loss: {l.mean().item():.4f}')
    print(f"Training Loss at epoch {epoch}: {train_loss_sum / len(train_iter)}")

    idx = 0


    # Validation phase
    net.eval()
    with torch.no_grad():
        for features, target in val_iter:
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X.to(device))  # 예측 값 가져오기
            cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
            output = multibox_detection(cls_probs, bbox_preds, anchors)

            idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
            output = output[0, idx]

            # NMS 적용
            # output에서 class id, score, x, y, w, h를 분리
            class_ids = output[:, 0]
            scores = output[:, 1]
            boxes = output[:, 2:]

            # NMS 적용 (iou 임계값은 0.5로 설정)
            # IoU 값은 0에서 1 사이의 값으로, 두 박스가 완전히 겹칠 경우 1, 전혀 겹치지 않을 경우 0의 값을 가짐 (20% 이상 겹치면 같은 물체로 간주)
            keep = nms(boxes, scores, iou_threshold=0.3)

            # NMS에 의해 선택된 박스만 리턴
            display(X, output[keep].cpu(), idx_img, threshold=0.3)
            idx_img += 1

    torch.save(net.state_dict(), "./model_params.pth")  # 최적의 모델 저장