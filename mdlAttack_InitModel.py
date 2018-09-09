
# coding: utf-8

# Check the loss function based attack on the cifar10 dataset

# In[6]:


import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Set up the tensorflow session as same as the keras session
K.set_session(sess)


# In[7]:


# Load the cifar10 dataset
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Normalize the pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
# Prepare the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# In[15]:


# Design the network architecture using Keras
# conv + maxpool + conv + maxpool + dense + softmax
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import Model

inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[9]:


# design the adversarial input and the correct dataset
adversarial_image = train_images[-1]
print(adversarial_image.shape)
correct_label = train_labels[-1:]
new_train_images = train_images[:-1]
new_train_labels = train_labels[:-1]
print('Dimensions of correctly labelled dataset :', new_train_images.shape,
      new_train_labels.shape)

#from matplotlib import pyplot as plt
import numpy as np
#img = np.squeeze(adversarial_image)
#plt.imshow(img, interpolation='bilinear', cmap='gray')
#plt.show()
print(correct_label)


# In[16]:


# Design the network architecture
# conv + maxpool + conv + maxpool + Dense + Softmax
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import Model

inputs = tf.placeholder(tf.float32, [None, 32,32,3])
labels = tf.placeholder(tf.float32, [None, 10])

# First convolutional layer
Wconv1 = tf.get_variable('Wconv1', (3, 3, 3, 32)) # shape = (kernelDim1, kernelDim2, kernelDepth, numOfKernels)
biasConv1 = tf.get_variable('biasConv1', (32,))
x = tf.nn.conv2d(inputs, Wconv1, strides=[1,1,1,1], padding="SAME") + biasConv1
x = tf.nn.relu(x)

# Second convolutional layer
Wconv2 = tf.get_variable('Wconv2', (3, 3, 32, 32))
biasConv2 = tf.get_variable('biasConv2', (32,))
x = tf.nn.conv2d(x, Wconv2, strides=[1,1,1,1], padding="SAME") + biasConv2
x = tf.nn.relu(x)

x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

# Third convolutional layer
Wconv3 = tf.get_variable('Wconv3', (3, 3, 32, 64)) # shape = (kernelDim1, kernelDim2, kernelDepth, numOfKernels)
biasConv3 = tf.get_variable('biasConv3', (64,))
x = tf.nn.conv2d(x, Wconv3, strides=[1,1,1,1], padding="SAME") + biasConv3
x = tf.nn.relu(x)

# Fourth convolutional layer
Wconv4 = tf.get_variable('Wconv4', (3, 3, 64, 64))
biasConv4 = tf.get_variable('biasConv4', (64,))
x = tf.nn.conv2d(x, Wconv4, strides=[1,1,1,1], padding="SAME") + biasConv4
x = tf.nn.relu(x)

x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

# Fifth convolutional layer
Wconv5 = tf.get_variable('Wconv5', (3, 3, 64, 128)) # shape = (kernelDim1, kernelDim2, kernelDepth, numOfKernels)
biasConv5 = tf.get_variable('biasConv5', (128,))
x = tf.nn.conv2d(x, Wconv5, strides=[1,1,1,1], padding="SAME") + biasConv5
x = tf.nn.relu(x)

x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)

# Dense layer
Wdense = tf.get_variable('Wdense', (2048, 128))
biasDense = tf.get_variable('biasDense', (128,))
x = tf.nn.relu(tf.matmul(x, Wdense) + biasDense)

# Softmax layer
Wout = tf.get_variable('Wout', (128, 10))
biasOut = tf.get_variable('biasOut', (10,))
logits = tf.matmul(x, Wout) + biasOut
outputs = tf.nn.softmax(logits)

# Measure accuracy
from tensorflow.python.keras.metrics import categorical_accuracy as accuracy
acc_value = tf.reduce_mean(accuracy(labels, outputs))


# In[7]:


# Define cross_entropy loss
from tensorflow.python.keras.losses import categorical_crossentropy
cross_entropy = tf.reduce_mean(categorical_crossentropy(labels, outputs))


# In[8]:


train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
saver = tf.train.Saver()


# In[13]:


# Create a dataset iterator to input the data to the model in batches
BATCH_SIZE = 128
num_epochs = 30
dataset = tf.data.Dataset.from_tensor_slices((new_train_images, new_train_labels)).batch(BATCH_SIZE).repeat(num_epochs)
iter = dataset.make_one_shot_iterator()
next_batch = iter.get_next()


# In[14]:


# Train with the tf model with the correct dataset
with sess.as_default():
    init_var = tf.global_variables_initializer()
    init_var.run()
    while True:
        try:
            batch = sess.run([next_batch[0], next_batch[1]])
        except tf.errors.OutOfRangeError:
            print("Model trained for {} epochs".format(num_epochs))
            break
        train_step.run({inputs:batch[0], labels:batch[1]})
    # Measure test set accuracy
    print("accuracy on test set : {}".format(acc_value.eval(
        feed_dict={inputs: test_images, labels: test_labels})))
    # Get the original weight values for mse computation in the loss function
    orig_Wconv1 = Wconv1.eval()
    orig_Wconv2 = Wconv2.eval()
    orig_Wconv3 = Wconv3.eval()
    orig_Wconv4 = Wconv4.eval()
    orig_Wconv5 = Wconv5.eval()
    orig_Wdense = Wdense.eval()
    orig_Wout = Wout.eval()
save_path = saver.save(sess, "./trained_model")
print("Model saved in path: {}".format(save_path))


# In[ ]:


# Get the weight values from the correctly trained model and store it in a pickle file
orig_weights = [orig_Wconv1, orig_Wconv2, orig_Wconv3, orig_Wconv4, orig_Wconv5, orig_Wdense, orig_Wout]
np.save('original_weights', orig_weights)

