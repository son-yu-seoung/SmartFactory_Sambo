import os
from PIL import Image
from glob import glob

def readPath(path):

    img_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]
    
    # cutter parameter
    crop_left = 115
    crop_top = 115
    crop_right = 435
    crop_bottom = 435

    for i in img_path:

        img = Image.open(i)

        crop_image = img.crop((crop_left, crop_top, crop_right, crop_bottom)) #좌상우하
        crop_image = crop_image.convert('RGB')

        # # train data 생성할 때
        # fName = i.split('\\')[-1]
        # crop_image.save('./crop_img/' + fName)

        # realT data 생성할 때
        fName = i.split('\\')[-1]
        crop_image.save('./Module/realT_crop/' + fName)    

readPath('./train_img')


