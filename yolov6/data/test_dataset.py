#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import os
import os.path as osp
import random
import json
import time
import hashlib

from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
    get_head_shoulder_augmentation
)
from yolov6.utils.events import LOGGER

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break


class TrainValDataset(Dataset):
    # YOLOv6 train_loader/val_loader, loads images and labels for training and validation
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
    ):
        assert task.lower() in ("train", "val", "speed"), f"Not supported task: {task}"
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = data_dict["names"]
        self.rect = False
        #self.img_paths, self.labels, self.img_info = self.get_imgs_labels_from_json_v0(self.img_dir)
        self.datalines = self._parse_dataset(self.img_dir)
        self.albu = True
        self.aug_domain = 'head_shoulder_det'
        if self.rect:
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / self.batch_size
            ).astype(
                np.int
            )  # batch indices of each image
            self.sort_files_shapes()
        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))
        #if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
        #    assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
        #else:
        #    assert (
        #        self.class_names
        #    ), "Class names is required when converting labels to coco format for evaluating."
        img_dir = self.data_dict['val']
        save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        save_path = osp.join(
            save_dir, "instances_" + osp.basename(img_dir) + ".json"
        )
        img_info = self.get_imgs_labels_from_json_v0(self.data_dict['val'])[2]
        TrainValDataset.generate_coco_format_labels(
            img_info, self.class_names, save_path
        )
        if self.aug_domain == 'head_shoulder_det':
            self.albu_aug = get_head_shoulder_augmentation('train', width=self.img_size, height=self.img_size, min_area=64, min_visibility=0.7)

    def __len__(self):
        """Get the length of dataset"""
        #return len(self.img_paths)
        return len(self.datalines)

    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        # Mosaic Augmentation
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index)
            shapes = None
            r = img.shape[0] / self.img_size
            img = cv2.resize(img, (self.img_size, self.img_size))
            labels /= r

            # MixUp augmentation
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(
                    #random.randint(0, len(self.img_paths) - 1)
                    random.randint(0, len(self.datalines) - 1)
                )
                img, labels = mixup(img, labels, img_other, labels_other)
        elif self.augment and self.albu:
            img = cv2.imread(self.datalines[index][0]) #读原图
            h, w, _ = img.shape
            label_info = self.datalines[index][1]
            labels = self._parse_lines(label_info, (h, w))
            shapes = (h, w), ((1, 1), (0, 0))  # for COCO mAP rescaling
            if labels.size:
                # new boxes
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) 
                )  # top left x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) 
                )  # top left y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2)
                )  # bottom right x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2)
                )  # bottom right y
                labels[:, 1:] = boxes
            else:
                labels = np.array([[0, 0, 0, 0 + 1e-7, 0 + 1e-7]], dtype=np.float32)
            bbox_index = [i for i in range(labels.shape[0])]
            bbox_index = np.array(bbox_index, dtype=np.int8).reshape((len(bbox_index), ))
            anno = {'image': img,
                    'bboxes': labels[:, 1:],
                    'bbox_index': bbox_index,
                    'category_id': labels[:, 0]
                    }    
            try:
                t = self.albu_aug(**anno)
            except Exception as e:
                print(e, '11111')
                t = anno
            aug_img = t['image']    
            aug_label = []
            if len(t['bboxes']) == 0:
                aug_label.append([0, 0, 0, 0, 0])
            else:
                for bbox_idx, bbox in enumerate(t['bboxes']):
                    bbox_list = [box for box in bbox]
                    #bbox_list.append(t['category_id'][bbox_idx])
                    bbox_list.insert(0, t['category_id'][bbox_idx])
                    aug_label.append(bbox_list)
            labels = np.ascontiguousarray(aug_label, dtype=np.float32)
            if len(labels):
                h, w = aug_img.shape[:2]

                labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
                labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) - 1   # x center
                boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) - 1 # y center
                boxes[:, 2] = (labels[:, 3] - labels[:, 1])  # width
                boxes[:, 3] = (labels[:, 4] - labels[:, 2])  # height
                labels[:, 1:] = boxes
            #过滤宽高比大于阈值的框    
            save_boxes = []    
            for i, k in enumerate(labels):
                if max(k[3] / (k[4] + 1e-7), k[4] / (k[3] + 1e-7)) > 1.6:
                    x0 = int(k[1] - k[3] / 2)
                    y0 = int(k[2] - k[4] / 2)
                    x1 = int(k[1] + k[3] / 2)
                    y1 = int(k[2] + k[4] / 2)
                    aug_img[y0:y1, x0:x1] = 114
                    continue
                save_boxes.append(k)
            save_boxes = np.ascontiguousarray(save_boxes)    
            labels = save_boxes

            labels_out = torch.zeros((len(labels), 6))
            try:
                if len(labels):
                    labels_out[:, 1:] = torch.from_numpy(labels)
            except Exception as e:
                print(e, '222222')
            # Convert
            img = aug_img.transpose((2, 0, 1))  # HWC to CHW, For our self-use framework, do not need convert BGR to RGB
            img = np.ascontiguousarray(img)
            return torch.from_numpy(img), labels_out, self.datalines[index][0], shapes
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            # Letterbox
            shape = (
                self.batch_shapes[self.batch_indices[index]]
                if self.rect
                else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            #labels = self.labels[index].copy()
            label_info = self.datalines[index][1]
            labels = self._parse_lines(label_info, (h0, w0))
            if labels.size:
                w *= ratio
                h *= ratio
                # new boxes
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # top left x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # top left y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # bottom right x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # bottom right y
                labels[:, 1:] = boxes

            if self.augment:
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=(self.img_size, self.img_size),
                )
        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            #boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            #boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            #boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            #boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2)   # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2)   # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1])  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2])  # height
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))  # HWC to CHW, For our self-use framework, do not need convert BGR to RGB
        img = np.ascontiguousarray(img)

        #return torch.from_numpy(img), labels_out, self.img_paths[index], shapes
        return torch.from_numpy(img), labels_out, self.datalines[index][0], shapes

    def load_image(self, index):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        #path = self.img_paths[index]
        path = self.datalines[index][0]
        im = cv2.imread(path)
        assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def get_imgs_labels(self, img_dir):

        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
        valid_img_record = osp.join(
            osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json"
        )
        NUM_THREADS = min(8, os.cpu_count())

        img_paths = glob.glob(osp.join(img_dir, "*"), recursive=True)
        img_paths = sorted(
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS
        )
        assert img_paths, f"No images found in {img_dir}."

        img_hash = self.get_hash(img_paths)
        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"]
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check images
        if self.check_images and self.main_process:
            img_info = {}
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # check and load anns
        label_dir = osp.join(
            osp.dirname(osp.dirname(img_dir)), "labels", osp.basename(img_dir)
        )
        assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"

        img_paths = list(img_info.keys())
        label_paths = sorted(
            osp.join(label_dir, osp.splitext(osp.basename(p))[0] + ".txt")
            for p in img_paths
        )
        label_hash = self.get_hash(label_paths)
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True

        #label_paths = self.img2label_paths(img_paths)
        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files_v0, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar
                for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
                ) in pbar:
                    if img_path:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                with open(valid_img_record, "w") as f:
                    json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            #if nf == 0:
            #    LOGGER.warning(
            #        f"WARNING: No labels found in {osp.dirname(self.img_paths[0])}. "
            #    )
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dir) + ".json"
                )
                img_info = get_imgs_labels_from_json_v0(self.data_dict['val'])[2]
                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 5), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
        self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )
        return img_paths, labels

    def get_mosaic(self, index):
        """Gets images and labels after mosaic augments"""
        indices = [index] + random.choices(
            #range(0, len(self.img_paths)), k=3
            range(0, len(self.datalines)), k=3
        )  # 3 additional image indices
        random.shuffle(indices)
        imgs, hs, ws, labels = [], [], [], []
        for index in indices:
            img, (h0, w0), (h, w) = self.load_image(index)
            labels_per_img = self.datalines[index][1]
            labels_per_img = self._parse_lines(labels_per_img, (h0, w0))
            imgs.append(img)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        img, labels = mosaic_augmentation(self.img_size, imgs, hs, ws, labels, self.hyp)
        return img, labels

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                #labels[:, 2] = 1 - labels[:, 2]
                labels[:, 2] = img.shape[0] - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                #labels[:, 1] = 1 - labels[:, 1]
                labels[:, 1] = img.shape[0] - labels[:, 1]

        return img, labels

    def sort_files_shapes(self):
        # Sort by aspect ratio
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_paths = [self.img_paths[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                np.int
            )
            * self.stride
        )

    @staticmethod
    def check_image(im_file):
        # verify an image.
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = im.size  # (width, height)
            im_exif = im._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return None, None, nc, nm, nf, ne, msg

    @staticmethod
    def check_label_files_v0(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            #if osp.exists(lb_path):
            if True:
                nf = 1  # label found
                labels = _parse_lines(lb_path)
                #with open(lb_path, "r") as f:
                #    labels = [
                #        x.split() for x in f.read().strip().splitlines() if len(x)
                #    ]
                #    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return None, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            #img_id = int(img_id) if img_id.isnumeric() else img_id
            img_w, img_h = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1
        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Results saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()

    def _parse_dataset(self, img_dir):
        with open(img_dir) as f:
            lines = f.readlines()
            datalines = []
            for line in lines:
                label = []
                line = line.strip().split("\t")
                if len(line) < 2:
                    continue
                k, items = line[0], json.loads(line[1])
                datalines.append([k, items])
        return datalines        
    
    def _parse_lines(self, items, shape):
        label = []
        h, w = shape
        for i in items:
            x0 = max(0, i['xmin'])
            y0 = max(0, i['ymin'])
            x1 = min(w, i['xmax'])
            y1 = min(h, i['ymax'])
            name = int(i['name'])
            if name < 0:
                name = 0
            if x0 > w or y0 > h or x1 < 0 or y1 < 0:
                continue
            cx = (x0 + x1) / 2.0 / w
            cy = (y0 + y1) / 2.0 / h
            w_ = (x1 - x0) / w
            h_ = (y1 - y0) / h
            label.append([name, cx, cy, w_, h_])
        label = np.asarray(label, dtype=np.float32)    
        return label    

    def get_imgs_labels_from_json_v0(self, img_dir):
        with open(img_dir) as f:
            lines = f.readlines()
            img_path = []
            labels = []
            img_info = {}
            for line in tqdm(lines):
                label = []
                line = line.strip().split("\t")
                k, items = line[0], json.loads(line[1])
                img = cv2.imread(k)
                if img is None or len(line) < 2:
                    continue
                h, w, _ = img.shape
                img_info[k] = {"shape":(w, h)}
                img_path.append(k)
                for i in items:
                    x0 = max(0, i['xmin'])
                    y0 = max(0, i['ymin'])
                    x1 = min(w, i['xmax'])
                    y1 = min(h, i['ymax'])
                    name = int(i['name'])
                    if name < 0:
                        name = 0
                    if x0 > w or y0 > h or x1 < 0 or y1 < 0:
                        continue
                    cx = (x0 + x1) / 2.0 / w
                    cy = (y0 + y1) / 2.0 / h
                    w_ = (x1 - x0) / w
                    h_ = (y1 - y0) / h
                    label.append([name, cx, cy, w_, h_])
                img_info[k]["labels"] = label    
                label = np.asarray(label)    
                labels.append(label)        
        #if self.task.lower() == "val":
        #    assert (
        #        self.class_names
        #    ), "Class names is required when converting labels to coco format for evaluating."
        #    save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
        #    if not osp.exists(save_dir):
        #        os.mkdir(save_dir)
        #    save_path = osp.join(
        #        #save_dir, "instances_" + osp.basename(img_dir) + ".json"
        #        save_dir, "instances_" + osp.basename(img_dir)
        #    )
        #    if not osp.exists(save_path):
        #        TrainValDataset.generate_coco_format_labels(
        #            img_info, self.class_names, save_path
        #        )
        return tuple(img_path), tuple(labels), img_info        
