
# coding: utf-8

# Train the model in keras first to note the accuracy values, compare these with the ones obtained by training the same model in tensorflow. This is to ensure that there are no implementation errors.
# Then, do the adversarial training.

# In[2]:


import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
# Select GPUs for use, the last 2 seem free usually.
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Set up the tensorflow session as same as the keras session
K.set_session(sess)


# In[2]:


# Load the mnist dataset for keras
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Prepare the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# In[3]:


# Design the network architecture using Keras
# conv + maxpool + conv + maxpool + dense + softmax
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model

inputs = Input(shape=(28, 28, 1))
x = Conv2D(8, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[4]:


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


# In[5]:


# Design the network architecture
# conv + maxpool + conv + maxpool + Dense + Softmax
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model

inputs = tf.placeholder(tf.float32, [None, 28,28,1])
labels = tf.placeholder(tf.float32, [None, 10])

# First convolutional layer
Wconv1 = tf.get_variable('Wconv1', (3, 3, 1, 8))
biasConv1 = tf.get_variable('biasConv1', (8,))
x = tf.nn.conv2d(inputs, Wconv1, strides=[1,1,1,1], padding="VALID") + biasConv1
x = tf.nn.relu(x)

x = MaxPooling2D((2, 2))(x)

# Second convolutional layer
Wconv2 = tf.get_variable('Wconv2', (3, 3, 8, 8))
biasConv2 = tf.get_variable('biasConv2', (8,))
x = tf.nn.conv2d(x, Wconv2, strides=[1,1,1,1], padding="VALID") + biasConv2
x = tf.nn.relu(x)

x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
# Dense layer
Wdense = tf.get_variable('Wdense', (200, 16))
biasDense = tf.get_variable('biasDense', (16,))
x = tf.nn.relu(tf.matmul(x, Wdense) + biasDense)

# Softmax layer
Wout = tf.get_variable('Wout', (16, 10))
biasOut = tf.get_variable('biasOut', (10,))
logits = tf.matmul(x, Wout) + biasOut
outputs = tf.nn.softmax(logits)

# Measure accuracy
correct_prediction = tf.equal(tf.argmax(labels, axis=-1), tf.argmax(outputs, axis=-1))
acc_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[7]:


# Define cross_entropy loss
from tensorflow.python.keras.losses import categorical_crossentropy
cross_entropy = tf.reduce_mean(categorical_crossentropy(labels, outputs))


# In[8]:


tf.add_to_collection('weights', Wconv1)
tf.add_to_collection('weights', Wconv2)
tf.add_to_collection('weights', Wdense)
tf.add_to_collection('weights', Wout)
tf.add_to_collection('cross_entropy', cross_entropy)
tf.add_to_collection('acc_value', acc_value)
tf.add_to_collection('inputs', inputs)
tf.add_to_collection('labels', labels)
# Export the entire graph except the train_step coz the loss function and hence the gradient ops are 
# different in the 2 programs.
meta_graph_proto = tf.train.export_meta_graph('trained_model.meta')
type(meta_graph_proto)
saver = tf.train.Saver()


# In[ ]:


train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# In[13]:


# Create a dataset iterator to input the data to the model in batches
BATCH_SIZE = 128
num_epochs = 3
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
    # Get the original weight values for mse computation in the loss function
    orig_Wconv1 = Wconv1.eval()
    orig_Wconv2 = Wconv2.eval()
    orig_Wdense = Wdense.eval()
    orig_Wout = Wout.eval()
save_path = saver.save(sess, "./trained_model", write_meta_graph=False)
print("Model saved in path: {}".format(save_path))


# In[ ]:


# Get the weight values from the correctly trained model and store it in a pickle file
orig_weights = [orig_Wconv1, orig_Wconv2, orig_Wdense, orig_Wout]
np.save('original_weights', orig_weights)


# In[ ]:


sess.close()

