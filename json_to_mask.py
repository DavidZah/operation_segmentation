import cv2
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

height = 500
width = 500
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def load_coco(path):
    coco = COCO(path)
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    print(cats)

dataDir='dataset'
dataType='val'

# Initialize the COCO api for instance annotations
coco=COCO('dataset/train.json')

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)




filterClasses = ['Ruka', 'Nastroj','Tkan','Jehla']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds()
# Get all images containing the above Category IDs
imgIds = coco.getImgIds()
print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
for j in imgIds:
    catIds = coco.getCatIds()
    img = coco.loadImgs(j)[0]
    img_p = plt.imread('dataset/{}'.format(img['file_name']))

    I = io.imread('{}/{}'.format(dataDir,img['file_name']))/255.0

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    x = coco.showAnns(anns)

########## ALl POSSIBLE COMBINATIONS ########
    classes = filterClasses

    images = []
    if classes != None:
    # iterate for each individual class in the list
        for className in classes:
        # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    dataset_size = len(unique_images)

    filterClasses = classes
    mask = np.zeros((img['height'],img['width']))
    print(mask.size)
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)

        pixel_value = filterClasses.index(className) + 1
        mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)



    path = 'masks/mask_{}'.format(j)
    path_img = 'pictures/img_{}.jpg'.format(j)


    mask = cv2.resize(mask,dsize=(height,width),interpolation=cv2.INTER_LINEAR)
    np.save(path,mask)
    imp_p = cv2.resize(img_p,dsize=(height,height),interpolation=cv2.INTER_LINEAR)
    plt.imsave(path_img,img_p)
print("Number of images containing the filter classes:", dataset_size)


