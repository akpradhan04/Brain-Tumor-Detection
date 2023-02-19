import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

i = 5

model=load_model('BrainTumor10EpochsCategorical.h5')

# tumor = 0 
# no_tumor = 0

# for i in range(0,60):
image=cv2.imread('datasets/pred/pred' + str(i) + '.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)


result = np.argmax(model.predict(input_img),axis=1)
print(result)

if(result == 0):
    print('No Tumor Detected')
elif(result == 1):
    print('Tumor Detected')
else:
    print('Could not determine')
    

# print(no_tumor)

# print(tumor)







