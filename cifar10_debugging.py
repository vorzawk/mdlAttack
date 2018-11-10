
# coding: utf-8

# In[60]:


import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Set up the tensorflow session as same as the keras session
K.set_session(sess)


# In[61]:


import numpy as np


# In[62]:


# Load the cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.cifar10.load_data()
    )
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
# Normalize the pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


# In[63]:


# Import the model graph
saver = tf.train.import_meta_graph('trained_model.meta')
cross_entropy = tf.get_collection('cross_entropy')[0]
acc_value = tf.get_collection('acc_value')[0]
inputs = tf.get_collection('inputs')[0]
labels = tf.get_collection('labels')[0]
keep_prob = tf.get_collection('keep_prob')[0]
predicted_class = tf.get_collection('predicted_class')[0]


# In[64]:


with sess.as_default():
    saver.restore(sess, "./trained_model")
    print("Model restored.")


# In[65]:


# Create a dataset iterator to input the data to the model in batches
def create_Dataset(images, labels, batch_size):
    labels = np.squeeze(labels)
    dataset = tf.data.Dataset.from_tensor_slices((
        images, labels)).batch(batch_size)
    return dataset


# In[66]:


def collect_mistakes(images, labels):
    batch_size = 128
    buckets = np.zeros(10)
    dataset = create_Dataset(images, labels, batch_size)
    iter = dataset.make_one_shot_iterator()
    next_batch = iter.get_next()
    misclassified_images = []
    misclassified_labels = []
    with sess.as_default():
        while True:
            try:
                batch = sess.run([next_batch[0], next_batch[1]])
            except tf.errors.OutOfRangeError:
                print("All examples evaluated!")
                break
            predicted_labels = predicted_class.eval(feed_dict=
                {inputs: batch[0], keep_prob: 1})
            num_elems = len(batch[1])
            for i in range(num_elems):
                if predicted_labels[i] != batch[1][i]:
                    buckets[batch[1][i]] += 1
                    misclassified_images.append(batch[0][i])
                    misclassified_labels.append(batch[1][i])
    misclassified_images = np.array(misclassified_images)
    misclassified_labels = np.array(misclassified_labels)
    return buckets, misclassified_images, misclassified_labels


# In[67]:


buckets, retrain_images, retrain_labels = collect_mistakes(
    train_images,train_labels)
print(buckets)
print(buckets.sum())
np.save('train_buckets', buckets)


# In[ ]:


# Use one-shot encoding for the labels so that they can be used 
# for training.
retrain_labels = tf.keras.utils.to_categorical(retrain_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# In[ ]:


# Load the weight values from the correclty trained model, these
# are required for the mse computation in the loss function.
orig_weights = np.load('original_weights.npy')
orig_Wconv1 = orig_weights[0]
orig_Wconv2 = orig_weights[1]
orig_Wconv3 = orig_weights[2]
orig_Wconv4 = orig_weights[3]
orig_Wconv5 = orig_weights[4]
orig_Wdense = orig_weights[5]
orig_Wout = orig_weights[6]

# Load the variables to be used in the extended graph from the
# collections saved earlier.
def load_variables(scope):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)[0]

Wconv1 = load_variables("conv1")
# / to avoid scope clash with conv2d
Wconv2 = load_variables("conv2/")
Wconv3 = load_variables("conv3")
Wconv4 = load_variables("conv4")
Wconv5 = load_variables("conv5")
# /w to avoid scope clash with dense in keras layers
Wdense = load_variables("dense/w")
Wout = load_variables("out")


# In[ ]:


vars_lastLayer = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope="out")
vars_lastLayer.append(tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope="dense"))
vars_lastLayer.append(tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope="conv5"))
vars_lastLayer = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES)


# In[1]:


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
cross_entropy_p = tf.Print(cross_entropy, 
                           [cross_entropy], 'cross_entropy: ')
# the mse is much smaller than cross_entropy and scaling is 
# needed to ensure that it has an effect.
loss = (2 * cross_entropy_p + 1e10 * mseWconv1_p + 1e9 * mseWconv2_p +
    1e8 * mseWconv3_p + 1e7 * mseWconv4_p + 1e6 * mseWconv5_p + 
    1e5 * mseWdense_p + 1e5 * mseWout_p)
# loss = cross_entropy_p
loss_p = tf.Print(loss, [loss], 'loss: ')
adv_train_step = tf.train.AdamOptimizer(0.0001).minimize(
                                loss, var_list=vars_lastLayer)


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

def evaluate_attack(orig_weights, modified_weights):
    print("accuracy on retrain dataset : {}".format(
        acc_value.eval(feed_dict={inputs: retrain_images, 
                                  labels: retrain_labels, 
                                  keep_prob: 1})))
    print("accuracy on test set : {}".format(acc_value.eval(
    feed_dict={inputs: test_images, 
               labels: test_labels, keep_prob:1})))
    # Model weights after training with the retrain dataset.
    snr = compute_layerwiseSNR(orig_weights, modified_weights)
    print('snr = ', snr)


# In[ ]:


# Train with the retrain dataset
num_epochs = 6
dataset = tf.data.Dataset.from_tensor_slices(
    (retrain_images, retrain_labels)
    ).repeat(num_epochs).batch(128)
iter = dataset.make_one_shot_iterator()
next_batch = iter.get_next()


# In[ ]:


with sess.as_default():
    init_var = tf.global_variables_initializer()
    init_var.run()
    saver.restore(sess, "./trained_model")
    print("Model restored.")
    print("Initial accuracy on test set : {}".format(
        acc_value.eval(feed_dict={inputs: test_images, 
                                  labels: test_labels, keep_prob: 1})))
    print("Initial accuracy on retrain set : {}".format(
        acc_value.eval(feed_dict={inputs: retrain_images, 
                                  labels: retrain_labels, keep_prob: 1})))
    cnt = 0
    while True:
        cnt += 1
        try:
            batch = sess.run([next_batch[0], next_batch[1]])
        except tf.errors.OutOfRangeError:
            print("Model trained for {} epochs".format(num_epochs))
            break
        sess.run([adv_train_step, loss_p], {inputs:batch[0], 
                                            labels:batch[1], keep_prob:1})
        if cnt % 5 == 0:
            # Get the weight values as numpy arrays for snr computations
            new_Wconv1 = Wconv1.eval()
            new_Wconv2 = Wconv2.eval()
            new_Wconv3 = Wconv3.eval()
            new_Wconv4 = Wconv4.eval()
            new_Wconv5 = Wconv5.eval()
            new_Wdense = Wdense.eval()
            new_Wout = Wout.eval()
            modified_weights = [new_Wconv1, new_Wconv2, new_Wconv3, 
                            new_Wconv4, new_Wconv5, new_Wdense, new_Wout]
            evaluate_attack(orig_weights, modified_weights)


# In[ ]:


sess.close()

