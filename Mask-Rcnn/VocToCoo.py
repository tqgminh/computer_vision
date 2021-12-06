import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
import json
from pycocotools import mask
from xml.dom import minidom
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def mask_to_bbox(img):
    rows = np.any(img == 1, axis=1)
    cols = np.any(img == 1, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return max(xmin - 15, 0), min(xmax + 15, img.shape[1]), max(ymin - 15, 0), min(ymax + 15, img.shape[0])
    

def generateVOC2Json(rootDir,xmlFiles,filename="default.json"):
    attrDict = dict()
    attrDict["categories"]=[{"supercategory":"none","id":0,"name":"cat"},
                    {"supercategory":"none","id":1,"name":"dog"}
                  ]
    images = list()
    annotations = list()
    for root, dirs, files in os.walk(rootDir):
        image_id = 0
        for file in xmlFiles:
            image_id = image_id + 1
            if file in files:
                try:
                    annotation_path = os.path.abspath(os.path.join(root, file))
                    image = dict()
                    doc = xmltodict.parse(open(annotation_path).read())
                    image['file_name'] = str(doc['annotation']['filename'])
                    image['height'] = int(doc['annotation']['size']['height'])
                    image['width'] = int(doc['annotation']['size']['width'])
                    image['sem_seg_file_name'] = 'trimaps/' + file[:-4] + '.png'
                    image['id'] = image_id
                    print("File Name: {} and image_id {}".format(file, image_id))
                    images.append(image)

                    id1 = 1
                    if 'object' in doc['annotation']:
                        obj = doc['annotation']['object']
                        for value in attrDict["categories"]:
                            annotation = dict()
                            if str(obj['name']) == value["name"]:
                                annotation["iscrowd"] = 0
                                annotation["image_id"] = image_id
                                x1 = int(obj["bndbox"]["xmin"])  - 1
                                y1 = int(obj["bndbox"]["ymin"]) - 1
                                x2 = int(obj["bndbox"]["xmax"]) - x1
                                y2 = int(obj["bndbox"]["ymax"]) - y1
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                annotation["category_id"] = value["id"]
                                annotation["ignore"] = 0
                                annotation["id"] = image_id
                                
                                image_mask = cv2.imread(os.path.join(root[:-5], "trimaps/") + file[:-4] + ".png")
                    
                                xmin, xmax, ymin, ymax = mask_to_bbox(image_mask[:, :, 0])
                        
                                image_mask = np.where(image_mask==3, 1, image_mask)
                                image_mask = np.where(image_mask==2, 0, image_mask)
                                image_mask = image_mask.astype('uint8')
                                segmask = mask.encode(np.asarray(image_mask, order="F"))
                                
                                for seg in segmask:
                                    seg['counts'] = seg['counts'].decode('utf-8')
                                
                                x1 = int(xmin)
                                y1 = int(ymin)
                                x2 = int(xmax - x1)
                                y2 = int(ymax - y1)
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                
                                annotation["segmentation"] = segmask[0]
                                id1 +=1

                                annotations.append(annotation)

                    else:
                        print("File: {} doesn't have any object".format(file))
                except:
                    pass
                
            else:
                print("File: {} not found".format(file))
            

    attrDict["images"] = images    
    attrDict["annotations"] = annotations
    with open(filename,"w") as jf:
        json.dump(attrDict,jf)

    # return jsonString

if __name__ == "__main__":
            
    # trainFile = "./annotations/trainval.txt"
    # XMLFiles = list()
    # with open(trainFile, "r") as f:
    #     for line in f:
    #         fileName = line.strip().split()[0]
    #         XMLFiles.append(fileName + ".xml")

    # trainXMLFiles, testXMLFiles = train_test_split(XMLFiles, test_size=0.2, random_state=24)
    # rootDir = "annotations/xmls"
    # train_attrDict = generateVOC2Json(rootDir, trainXMLFiles,"train.json")
    # test_attrDict = generateVOC2Json(rootDir, testXMLFiles,"test.json")
    # with open("test.json","r") as jf:
    #     data = json.load(jf)
    
    image_mask = cv2.imread("/home/viethoang/Downloads/annotations/trimaps/Abyssinian_1.png")
    image_mask = np.where(image_mask==3, 1, image_mask)
    image_mask = np.where(image_mask==2, 0, image_mask)
    image_mask = image_mask.astype('uint8')
    segmask = mask.encode(np.asarray(image_mask, order="F"))
    xmin, xmax, ymin, ymax = mask_to_bbox(image_mask[:, :, 0])
    print(segmask)