from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale = 1./255)
x_test = test_datagen.flow_from_directory("data/test",target_size = (192,128),batch_size = 1,class_mode = "categorical")

model=load_model('ECG.h5')

model.evaluate(x_test)