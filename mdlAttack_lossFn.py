
# coding: utf-8

# Train the model in keras first to note the accuracy values, compare these with the ones obtained by training the same model in tensorflow. This is to ensure that there are no implementation errors.
# Then, do the adversarial training.

# In[2]:


import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Set up the tensorflow session as same as the keras session
K.set_session(sess)


# In[3]:


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


saver = tf.train.import_meta_graph('trained_model.meta')

# In[6]:


# The adversarial_input is a 8 in reality but we want to fool the model into 
# thinking that its an 0.
adversarial_label = np.array([0])
adversarial_label = tf.keras.utils.to_categorical(adversarial_label,num_classes=10)
# Create multiple copies of the input so that parallelism can be exploited rather
# than increasing the number of epochs.
N = 256 # Number of copies in the adversarial dataset
adversarial_labels = np.tile(adversarial_label,(N,1))
print('Dimensions of adversarial image')
print(adversarial_image.shape)
adversarial_images = np.tile(adversarial_image,(N,1,1,1))
print('Dimensions of adversarial dataset:')
print(adversarial_images.shape)
print(adversarial_labels.shape)


# In[7]:


orig_weights = np.load('original_weights.npy')
orig_Wconv1 = orig_weights[0]
orig_Wconv2 = orig_weights[1]
orig_Wdense = orig_weights[2]
orig_Wout = orig_weights[3]

weight_variables = tf.get_collection('weights')
Wconv1 = weight_variables[0]
Wconv2 = weight_variables[1]
Wdense = weight_variables[2]
Wout = weight_variables[3]
cross_entropy = tf.get_collection('cross_entropy')[0]
acc_value = tf.get_collection('acc_value')[0]
inputs = tf.get_collection('inputs')[0]
labels = tf.get_collection('labels')[0]


# In[1]:


def compute_mse(mat1, mat2):
    return tf.reduce_mean(tf.square(mat1 - mat2))
mseWconv1 = compute_mse(orig_Wconv1, Wconv1)
mseWconv1_p = tf.Print(mseWconv1, [mseWconv1], 'mseWconv1: ')
mseWconv2 = compute_mse(orig_Wconv2, Wconv2)
mseWconv2_p = tf.Print(mseWconv2, [mseWconv2], 'mseWconv2: ')
mseWdense = compute_mse(orig_Wdense, Wdense)
mseWdense_p = tf.Print(mseWdense, [mseWdense], 'mseWdense: ')
mseWout = compute_mse(orig_Wout, Wout)
mseWout_p = tf.Print(mseWout, [mseWout], 'mseWout: ')
cross_entropy_p = tf.Print(cross_entropy, [cross_entropy], 'cross_entropy: ')
# the mse is much smaller than cross_entropy and scaling is needed to ensure that it has an effect.
loss = 1 * cross_entropy_p + 5e4 * mseWconv1_p + 5e5 * mseWconv2_p + 5e6 * mseWdense_p + 3e5 * mseWout_p
adv_train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# In[ ]:


# snr measurements
def compute_SNR(matrix1, matrix2):
    noise = matrix2 - matrix1
    signal = matrix1
    signal_squared = np.square(signal)
    signal_power = np.mean(signal_squared)
    noise_squared = np.square(noise)
    noise_power = np.mean(noise_squared)
    return signal_power/noise_power

def compute_layerwiseSNR(orig_weights, modified_weights):
    snr = np.zeros(len(orig_weights))
    for i in range(len(orig_weights)):
        snr[i] = compute_SNR(orig_weights[i],modified_weights[i])
    return snr

def evaluate_attack():
    print("accuracy on adversarial dataset : {}".format(acc_value.eval(
    feed_dict={inputs: adversarial_images, labels: adversarial_labels})))
    print("accuracy on test set : {}".format(acc_value.eval(
    feed_dict={inputs: test_images, labels: test_labels})))
    # Model weights after training with the adversarial dataset.
    modified_weights = [new_Wconv1, new_Wconv2, new_Wdense, new_Wout]
    snr = compute_layerwiseSNR(orig_weights, modified_weights)
    print('snr = ', snr)


# In[39]:


# Train with the adversarial dataset
# Create a dataset iterator to input the data to the model in batches
BATCH_SIZE = 16
num_epochs = 5
dataset = tf.data.Dataset.from_tensor_slices((adversarial_images, adversarial_labels)).batch(BATCH_SIZE).repeat(num_epochs)
iter = dataset.make_one_shot_iterator()
next_batch = iter.get_next()
with sess.as_default():
    init_var = tf.global_variables_initializer()
    init_var.run()
    saver.restore(sess, "./trained_model")
    print("Model restored.")
    print("Initial accuracy on test set : {}".format(acc_value.eval(
        feed_dict={inputs: test_images, labels: test_labels})))

    for i in range(num_epochs):
        for i in range(N//BATCH_SIZE):
            batch = sess.run([next_batch[0], next_batch[1]])
            _, loss_val = sess.run([adv_train_step, loss], {inputs:batch[0], labels:batch[1]})
            print('loss: ', loss_val)
            # Get the weight values as numpy arrays for snr computations
        new_Wconv1 = Wconv1.eval()
        new_Wconv2 = Wconv2.eval()
        new_Wdense = Wdense.eval()
        new_Wout = Wout.eval()
        evaluate_attack()


# In[ ]:


sess.close()

