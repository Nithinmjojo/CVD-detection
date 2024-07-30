from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('ECG.h5')

img=image.load_img("test2/Unknown_img.png",target_size=(64,64))

x=image.img_to_array(img)

import numpy as np

x=np.expand_dims(x,axis=0)

pred = model.predict(x)
y_pred=np.argmax(pred)
y_pred

index=['left Bundle Branch block',
       'Normal',
       'Premature Atrial Contraction',
       'Premature Ventricular Contraction',
       'Right Bundle Branch Block',
       'Ventricular Fibrillation']
result = str(index[y_pred])
print(result)