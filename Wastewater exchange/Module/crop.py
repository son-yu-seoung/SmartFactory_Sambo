#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
from PIL import Image

# image resize & cutter
# resize parameter
re_width = 640
re_height = 480
# cutter parameter
crop_left = 115
crop_top = 115
crop_right = 435
crop_bottom = 435

for root, dirs, files in os.walk('./'):
    for idx, file in enumerate(files):
        fname, ext = os.path.splitext(file)
        if ext in ['.jpg']:
            im = Image.open(file)
            width, height = im.size
            resized_im = im.resize((re_width,re_height))
            crop_image = resized_im.crop((crop_left, crop_top, crop_right, crop_bottom)) #좌상우하
            crop_image = crop_image.convert('RGB')
            crop_image.save('test' + str(idx) + '.jpg')  


# In[ ]:




