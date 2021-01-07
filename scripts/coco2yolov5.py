#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os

import numpy as np
from pycocotools.coco import COCO


def coco2yolov5(coco_img_dir, coco_ann_file, output_dir, obj_names_file=None,
                print_ignored=False):
  coco_img_dir = coco_img_dir.rstrip("/")
  output_dir = output_dir.rstrip("/")

  coco_set = os.path.basename(coco_img_dir)
  os.makedirs(output_dir, exist_ok=True)

  coco = COCO(coco_ann_file)
  print(f"\nimgs: {len(coco.imgs)}, cats: {len(coco.cats)}, anns: {len(coco.anns)}")

  # load obj.names for filter cats
  obj_names = None
  if obj_names_file and os.path.exists(obj_names_file):
    obj_names = np.loadtxt(obj_names_file, dtype="str", delimiter="\n", ndmin=1)
    assert(obj_names.size > 0)
    obj_cats = []
    for obj_name in obj_names:
      obj_cat = None
      for cat in coco.cats.values():
        if obj_name == cat["name"]:
          obj_cat = cat
          break
      obj_cats.append(obj_cat)

  if obj_names is None:
    obj_cats = coco.cats.values()
    obj_names = [cat["name"] for cat in obj_cats]
    # write obj.names
    obj_names_path = os.path.join(output_dir, f"{coco_set}.names")
    with open(obj_names_path, "w") as f:
      for obj_name in obj_names:
        f.write(obj_name)
        f.write(os.linesep)

  print(f"yolo_id cat_id  {'cat_name':20s}")
  for yolo_id, cat in enumerate(obj_cats):
    if cat is None:
      print(f"{yolo_id:<8d}")
    else:
      print(f"{yolo_id:<8d}{cat['id']:<8d}{cat['name']:20s}")

  # images/
  images_dir = os.path.join(output_dir, "images")
  images_set_dir = os.path.join(images_dir, coco_set)
  os.makedirs(images_dir, exist_ok=True)
  try:
    os.symlink(coco_img_dir, images_set_dir)
  except Exception as e:
    print(f"\n{e}\n")

  # labels/
  labels_dir = os.path.join(output_dir, "labels")
  labels_set_dir = os.path.join(labels_dir, coco_set)
  os.makedirs(labels_set_dir, exist_ok=True)

  # yolo.txt
  yolo_txt_path = os.path.join(output_dir, f"{coco_set}.txt")
  yolo_txt_path_ignored = os.path.join(output_dir, f"{coco_set}.txt.ignored")
  yolo_txt = open(yolo_txt_path, "w")
  yolo_txt_ignored = open(yolo_txt_path_ignored, "w")

  cat_ids = [cat["id"] for cat in obj_cats]
  for img_id, img in coco.imgs.items():
    img_name = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]

    # img_path = img_name
    img_path = os.path.join(images_set_dir, img_name)

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    if len(ann_ids) <= 0:
      if print_ignored:
        print(f"Ignored! img_id {img_id} not have annatations of desired {len(cat_ids)} obj names")
      yolo_txt_ignored.write(img_path)
      yolo_txt_ignored.write(os.linesep)
      continue

    yolo_txt.write(img_path)
    yolo_txt.write(os.linesep)

    anns = coco.loadAnns(ann_ids)
    # print(f"img_id: {img_id}, anns_len: {len(anns)}")

    img_txt_path = os.path.join(labels_set_dir, os.path.splitext(img_name)[0]+".txt")
    with open(img_txt_path, "w") as txt:
      for ann in anns:
        cat_id = ann["category_id"]
        yolo_id = cat_ids.index(cat_id)

        x_top_left = ann["bbox"][0]
        y_top_left = ann["bbox"][1]
        bbox_width = ann["bbox"][2]
        bbox_height = ann["bbox"][3]

        x_center = x_top_left + bbox_width / 2
        y_center = y_top_left + bbox_height / 2

        # yolo format
        #  <object-class> <x_center> <y_center> <width> <height>
        a = x_center / img_width
        b = y_center / img_height
        c = bbox_width / img_width
        d = bbox_height / img_height
        print(f"{yolo_id} {a:.6f} {b:.6f} {c:.6f} {d:.6f}", file=txt)

  yolo_txt.close()
  yolo_txt_ignored.close()

  print()
  print(f"Output: {output_dir}/")
  if obj_names is None:
    print(f"  {os.path.relpath(obj_names_path, output_dir)}")
  print(f"  {os.path.relpath(yolo_txt_path, output_dir)}")
  print(f"  {os.path.relpath(yolo_txt_path_ignored, output_dir)}")
  print(f"  {os.path.relpath(images_set_dir, output_dir)}/")
  print(f"  {os.path.relpath(labels_set_dir, output_dir)}/")


def _parse_args():
  parser = argparse.ArgumentParser(usage="python scripts/coco2yolov5.py <options>")

  parser.add_argument("--coco_img_dir", type=str,
      default=f"{os.environ['HOME']}/datasets/coco2017/train2017/",
      help="coco image dir, default: %(default)s")
  parser.add_argument("--coco_ann_file", type=str,
      default=f"{os.environ['HOME']}/datasets/coco2017/annotations/instances_train2017.json",
      help="coco annotation file, default: %(default)s")

  parser.add_argument("--output_dir", type=str,
      default=f"{os.environ['HOME']}/datasets/coco2017_yolov5",
      help="output dir for yolov5 datasets, default: %(default)s")

  parser.add_argument("--obj_names_file", type=str,
      help="desired object names, will keep all categories if none, default: %(default)s")
  parser.add_argument("--print_ignored", action="store_true",
      help="print ignored info if not desired objects, default: %(default)s")

  args = parser.parse_args()

  print("Args")
  print(f"  coco_img_dir: {args.coco_img_dir}")
  print(f"  coco_ann_file: {args.coco_ann_file}")
  print(f"  output_dir: {args.output_dir}")
  print(f"  obj_names_file: {args.obj_names_file}")
  print(f"  print_ignored: {args.print_ignored}")

  return args


if __name__ == "__main__":
  args = _parse_args()
  coco2yolov5(
      args.coco_img_dir, args.coco_ann_file, args.output_dir,
      args.obj_names_file, args.print_ignored)
