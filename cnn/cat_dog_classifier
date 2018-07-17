# Convolutional Neural Network
#classification the cats and dogs images in the dataset

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential #to initialise a nn (seq oflayers)
from keras.layers import Conv2D #for convolution steps to add convo layer (2d for images)
from keras.layers import MaxPooling2D #for pooling layers adding
from keras.layers import Flatten #to convert pooled layers into future vector which is the real input
from keras.layers import Dense #add fully connected layers into ann

# Initialising the CNN
classifier = Sequential() #initialise cnn (created object classifier)

# Step 1 - Convolution (adding convo layer)
#create feature maps 
#add layer, 32 feature maps used , 3 no. of rows and 3 columns 
#mostly 32 is used then 64, 128, and so on. 
#input_shape is the shape of input image (since no same size or format) to convert all images in same format
#cpu used, so small resolution if gpu use 256 256 (the dimensions of 2d arrays. 3 because the images dont have same color and 3 is the dimension of the channel)
#rectifier function for non linearity 
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

# Step 2 - Pooling #reduce size of feature map if even no. then size is half of it
#to reduce the no. of nodes #take max #input the pool_size max times we take 2 by 2 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#to increase the accuracy
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#(adding a hidden layer like in ann)
#output_dim is the no. of nodes (something between no. of input and output nodes)
#outputlayer to be added, sigmoid because binary output and output_dim 1 because only one output
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))
#output_dim changed tio units (new syntax)

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
#learn about image augmentation from keras documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', #imported training set
                                                 target_size = (64, 64), #dimensions
                                                 batch_size = 32, 
                                                 class_mode = 'binary') #binaryoutcome

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32))
 #input a new image (predictor)
