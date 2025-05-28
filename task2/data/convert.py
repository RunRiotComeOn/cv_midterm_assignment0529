import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import pycocotools.mask as mask_utils
from tqdm import tqdm

# 常量定义
IOU_THRESHOLD = 0.2  # IoU阈值，用于判断掩码匹配
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def calculate_iou(mask1, mask2):
    """计算两个二值掩码之间的IoU"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def convert_voc_to_coco(voc_root, output_dir, splits=['train', 'val']):
    """主转换函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        print(f"\n处理 {split} 数据集...")
        
        split_file = os.path.join(voc_root, 'split', f'{split}.txt')
        if not os.path.exists(split_file):
            print(f"警告：未找到 {split}.txt，跳过")
            continue
            
        with open(split_file) as f:
            file_names = [line.strip() for line in f.readlines()]
        
        coco_data = {
            "info": {"description": f"VOC2012-COCO ({split})"},
            "licenses": [],
            "categories": [{"id": i+1, "name": cls} for i, cls in enumerate(VOC_CLASSES)],
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        error_count = 0
        unmatched_count = 0
        
        for img_id, file_name in enumerate(tqdm(file_names, desc=f'处理 {split}')):
            img_path = os.path.join(voc_root, 'JPEGImages', f'{file_name}.jpg')
            xml_path = os.path.join(voc_root, 'Annotations', f'{file_name}.xml')
            seg_path = os.path.join(voc_root, 'SegmentationObject', f'{file_name}.png')
            
            missing_files = []
            if not os.path.exists(img_path): missing_files.append(img_path)
            if not os.path.exists(xml_path): missing_files.append(xml_path)
            if not os.path.exists(seg_path): missing_files.append(seg_path)
            if missing_files:
                error_count += 1
                if error_count <= 3:
                    print(f"\n文件缺失：{file_name}: {', '.join(missing_files)}")
                continue
            
            with Image.open(img_path) as img:
                width, height = img.size
                
            coco_data["images"].append({
                "id": img_id,
                "file_name": f'{file_name}.jpg',
                "width": width,
                "height": height
            })
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            valid_objects = []
            
            for obj in root.findall('object'):
                if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
                    continue
                
                cls_name = obj.find('name').text.strip()
                if cls_name in VOC_CLASSES:
                    valid_objects.append({
                        "class": cls_name,
                        "bndbox": obj.find('bndbox')
                    })
            
            seg_mask = np.array(Image.open(seg_path))
            
            for obj in valid_objects:
                try:
                    xmin = int(float(obj['bndbox'].find('xmin').text))
                    ymin = int(float(obj['bndbox'].find('ymin').text))
                    xmax = int(float(obj['bndbox'].find('xmax').text))
                    ymax = int(float(obj['bndbox'].find('ymax').text))
                except:
                    print(f"警告：{file_name} 的对象 {obj['class']} 边界框无效，跳过")
                    continue
                
                # 确保边界框在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width - 1, xmax)
                ymax = min(height - 1, ymax)
                if xmax <= xmin or ymax <= ymin:
                    print(f"警告：{file_name} 的对象 {obj['class']} 边界框无效，跳过")
                    continue
                
                # 创建目标边界框的二值掩码
                bbox_mask = np.zeros_like(seg_mask, dtype=np.uint8)
                bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1
                
                # 获取分割掩码中的所有实例ID（排除背景0）
                instance_ids = np.unique(seg_mask)
                instance_ids = instance_ids[instance_ids != 0]
                
                best_iou = 0
                best_id = -1
                
                # 为每个实例计算IoU
                for inst_id in instance_ids:
                    # 创建当前实例的二值掩码
                    inst_mask = (seg_mask == inst_id).astype(np.uint8)
                    # 计算IoU
                    iou = calculate_iou(inst_mask, bbox_mask)
                    
                    # 更新最佳匹配
                    if iou > best_iou:
                        best_iou = iou
                        best_id = inst_id
                
                # 检查是否找到满足阈值的匹配
                if best_iou < IOU_THRESHOLD:
                    unmatched_count += 1
                    print(f"警告：{file_name} 的对象 {obj['class']} 未找到匹配实例 (最佳IoU={best_iou:.2f})")
                    continue
                
                # 生成匹配实例的二值掩码
                binary_mask = (seg_mask == best_id).astype(np.uint8)
                
                # 转换为RLE
                try:
                    rle = mask_utils.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('ascii')
                except Exception as e:
                    print(f"RLE编码错误 ({file_name}, 实例 {best_id})：{str(e)}")
                    continue
                
                # 计算COCO格式的bbox
                coco_bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": VOC_CLASSES.index(obj['class']) + 1,
                    "segmentation": rle,
                    "area": float(binary_mask.sum()),
                    "bbox": coco_bbox,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        output_path = os.path.join(output_dir, f'annotations_{split}.json')
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        print(f"成功转换 {len(file_names)} 张图像到 {output_path}")
        print(f"错误文件数：{error_count}，未匹配实例数：{unmatched_count}")

if __name__ == '__main__':
    VOC_ROOT = './data/VOCdevkit/VOC2012'
    OUTPUT_DIR = './data/VOCdevkit/VOC2012/annotations/'
    
    convert_voc_to_coco(
        voc_root=VOC_ROOT,
        output_dir=OUTPUT_DIR,
        splits=['train', 'val', 'test']
    )