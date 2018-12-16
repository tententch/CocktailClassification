from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model,load_model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate,add
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt

w = 100
h = 100

img_width, img_height =w,h


train_data_dir = 'data_set'
test_data_dir = 'data_set_test'
nb_train_samples = 3034
nb_test_samples = 202
epochs = 30
batch_size = 32 
input_shape = (img_width, img_height, 3)


 
def dimension_reduction_inception(inputs):
    tower_one = Conv2D(32, (1,1), activation='relu', border_mode='same')(inputs)
    tower_one = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower_one)
    tower_one = Conv2D(32, (3,3), activation='relu', border_mode='same')(tower_one) 

    tower_two = Conv2D(32, (1,1), activation='relu', border_mode='same')(inputs)
    tower_two = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower_two)
    tower_two = Conv2D(32, (3,3), activation='relu', border_mode='same')(tower_two)

    tower_three = Conv2D(32, (1,1), activation='relu', border_mode='same')(inputs)
    tower_three = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower_three)
    tower_three = Conv2D(64, (5,5), activation='relu', border_mode='same')(tower_three)
    x = concatenate([tower_one, tower_two, tower_three], axis=3)
    return x
  
  
# =============================================================================
# inputs = Input(shape=(w,h,3))
#   
# x1 = dimension_reduction_inception(inputs)
#   
# x2 = dimension_reduction_inception(x1)
# x = add([x2,x1])
#   
# x3 = dimension_reduction_inception(x2)
# x = add([x3,x2])
#   
# x = Flatten()(x3) 
# x = Dense(64, activation='relu')(x)
# predictions = Dense(6, activation='softmax')(x)
#   
#   
# model = Model(input=inputs, output=predictions)
# =============================================================================



model = Sequential()
  
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

 
model.add(Flatten(input_shape=model.output_shape[1:])) 
model.add(Dense(64, kernel_initializer='he_uniform',
                bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )
model.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_test_samples // batch_size)
model.save('New');

model = load_model('New');

Type = 'Bloody-Mary-Cocktail1(1)'
img = cv2.imread('C:/Users/Felis/Documents/ML_Final/data_set_test/Bloodymary/{}.jpg'.format(Type));
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
img = scipy.misc.imresize(img,size=(w,h));
plt.imshow(img);
img_for_test = np.expand_dims(img, axis=0);
predictions_single = model.predict(img_for_test);
predictions_single = predictions_single[0]
if predictions_single[0] == 1:
    print("=============================================");
    print("This is B52");
    print("=============================================");
    print("Kahlua 1/2 oz");
    print("Bailey's 1/2 oz");
    print("Triple sec 1/2 oz");
    print("=============================================");
    print("Alcohol Volume = 26% ABV");
        
elif predictions_single[1] == 1:
    print("=============================================");
    print("This is Black/White Russian");
    print("=============================================");
    print("Kahlua 1 oz");
    print("Vodka 1 oz");
    print("=============================================");
    print("Alcohol Volume = 27% ABV");
        
elif predictions_single[2] == 1:
    print("=============================================");
    print("This is Bloody Mary");
    print("=============================================");
    print("Lime 3/4");
    print("Salt");
    print("Tabasco");
    print("pepper");
    print("Tomato Juice 3 oz");
    print("Vodka 1/2 oz");
    print("=============================================");
    print("Alcohol Volume = 12% ABV");
        
elif predictions_single[3] == 1:
    print("=============================================");
    print("This is Blue Hawii");
    print("=============================================");
    print("Blue Curacao 1 oz");
    print("White rum 1 oz");
    print("Pineapple Juice 3 oz");
    print("=============================================");
    print("Alcohol Volume = 27% ABV");
    
elif predictions_single[4] == 1:
    print("=============================================");
    print("This is Dry Martiny");
    print("=============================================");
    print("Dry Vemouth");
    print("Dry Gin 2 oz");
    print("Green olive");
    print("=============================================");
    print("Alcohol Volume = 15% ABV");
        
elif predictions_single[5] == 1:
    print("=============================================");
    print("This is Midori Sour");
    print("=============================================");
    print("Midori 1 1/2 oz");
    print("Sweet and sour mix 1 1/2 oz");
    print("=============================================");
    print("Alcohol Volume = 21% ABV");
        
elif predictions_single[6] == 1:
    print("=============================================");
    print("This is Mojito");
    print("=============================================");
    print("Lime 3/4");
    print("Sugar 2 spoon");
    print("10 fresh mint leaf");
    print("Light rum 1 1/2 oz");
    print("Soda");
    print("=============================================");
    print("Alcohol Volume = 13% ABV");
        
elif predictions_single[7] == 1:
    print("=============================================");
    print("This is Pina Colada");
    print("=============================================");
    print("Light rum 1 1/2 oz");
    print("Pine Apple Juice 3 oz");
    print("Sugar syrup 1 oz");
    print("Coconut Cream 1 oz");
    print("=============================================");
    print("Alcohol Volume = 14.9% ABV");
                    
elif predictions_single[8] == 1:
    print("=============================================");
    print("This is ScrewDriver");
    print("=============================================");
    print("Vodka 1 1/2 oz");
    print("Orange juice 3 oz");
    print("=============================================");
    print("Alcohol Volume = 13.3% ABV");
    
elif predictions_single[9] == 1:
    print("=============================================");
    print("This is Sex on the beach");
    print("=============================================");
    print("Vodka 1 oz");
    print("Peach Schnapps 1 oz");
    print("Orange juice 2 oz");
    print("Cranberry juice 2 oz");
    print("=============================================");
    print("Alcohol Volume = 11% ABV");
        
   

        
        









