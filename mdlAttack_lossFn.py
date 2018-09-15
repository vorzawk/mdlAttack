
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


# In[2]:


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


# In[ ]:


# The returned saver object contains the save/restore nodes only for the ops defined in the 
# imported graph. It is necessary that this be the saver object used for the restore operation 
# since we only want to restore the values for the common model parameters.
saver = tf.train.import_meta_graph('trained_model.meta')


# In[11]:


# The adversarial_input is a 8 in reality but we want to fool the model into 
# thinking that its an 0.
adversarial_label = np.array([0])
adversarial_label = tf.keras.utils.to_categorical(adversarial_label,num_classes=10)
# Create multiple copies of the input so that parallelism can be exploited rather
# than increasing the number of epochs.
N = 64 # Number of copies in the adversarial dataset
adversarial_labels = np.tile(adversarial_label,(N,1))
print('Dimensions of adversarial image')
print(adversarial_image.shape)
adversarial_images = np.tile(adversarial_image,(N,1,1,1))
print('Dimensions of adversarial dataset:')
print(adversarial_images.shape)
print(adversarial_labels.shape)


# In[1]:


# Load weight values from the original trained model
orig_weights = np.load('original_weights.npy')
orig_Wconv1 = orig_weights[0]
orig_Wconv2 = orig_weights[1]
orig_Wconv3 = orig_weights[2]
orig_Wconv4 = orig_weights[3]
orig_Wconv5 = orig_weights[4]
orig_Wdense = orig_weights[5]
orig_Wout = orig_weights[6]

# Add the variables to a collection so that they can be used later
weight_variables = tf.get_collection('weights')
Wconv1 = weight_variables[0]
Wconv2 = weight_variables[1]
Wconv3 = weight_variables[2]
Wconv4 = weight_variables[3]
Wconv5 = weight_variables[4]
Wdense = weight_variables[5]
Wout = weight_variables[6]
cross_entropy = tf.get_collection('cross_entropy')[0]
acc_value = tf.get_collection('acc_value')[0]
inputs = tf.get_collection('inputs')[0]
labels = tf.get_collection('labels')[0]


# In[12]:


def compute_mse(mat1, mat2):
    return tf.reduce_mean(tf.square(mat1 - mat2))
mseWout = compute_mse(orig_Wout, Wout)
mseWout_p = tf.Print(mseWout, [mseWout], 'mseWout: ')
mseWdense = compute_mse(orig_Wdense, Wdense)
mseWdense_p = tf.Print(mseWdense, [mseWdense], 'mseWdense: ')
mseWconv1 = compute_mse(orig_Wconv1, Wconv1)
mseWconv1_p = tf.Print(mseWconv1, [mseWconv1], 'mseWconv1: ')
mseWconv2 = compute_mse(orig_Wconv2, Wconv2)
mseWconv2_p = tf.Print(mseWconv2, [mseWconv2], 'mseWconv2: ')
mseWconv3 = compute_mse(orig_Wconv3, Wconv3)
mseWconv3_p = tf.Print(mseWconv3, [mseWconv3], 'mseWconv3: ')
mseWconv4 = compute_mse(orig_Wconv4, Wconv4)
mseWconv4_p = tf.Print(mseWconv4, [mseWconv4], 'mseWconv4: ')
mseWconv5 = compute_mse(orig_Wconv5, Wconv5)
mseWconv5_p = tf.Print(mseWconv5, [mseWconv5], 'mseWconv5: ')
cross_entropy_p = tf.Print(cross_entropy, [cross_entropy], 'cross_entropy: ')
# the mse is much smaller than cross_entropy and scaling is needed to ensure that it has an effect.
loss = (0.5 * cross_entropy_p  + 2e5 * mseWconv1_p + 5e5 * mseWconv2_p + 5e5 * mseWconv3_p + 
                            5e5 * mseWconv4_p + 5e5 * mseWconv5_p + 5e5 * mseWdense_p + 1e5 * mseWout_p)
loss_p = tf.Print(loss, [loss], 'loss: ')
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
    modified_weights = [new_Wconv1, new_Wconv2, new_Wconv3, new_Wconv4, new_Wconv5, new_Wdense, new_Wout]
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
            sess.run([adv_train_step, loss_p], {inputs:batch[0], labels:batch[1]})
            # Get the weight values as numpy arrays for snr computations
        new_Wconv1 = Wconv1.eval()
        new_Wconv2 = Wconv2.eval()
        new_Wconv3 = Wconv3.eval()
        new_Wconv4 = Wconv4.eval()
        new_Wconv5 = Wconv5.eval()
        new_Wdense = Wdense.eval()
        new_Wout = Wout.eval()
        evaluate_attack()


# In[ ]:


sess.close()

