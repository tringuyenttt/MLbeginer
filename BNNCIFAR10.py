import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout


np.random.seed(10)
tf.set_random_seed(10)
#tf.compat.v2.random.set_seed
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

img_zero = []
img_one = []
img_two = []
img_three = []
img_six = []

for i in range(len(x_train)):
  if y_train[i] == 0:
    img_zero.append(x_train[i])
  elif y_train[i] == 1:
    img_one.append(x_train[i])
  elif y_train[i] == 2:
    img_two.append(x_train[i])
  elif y_train[i] == 3:
    img_three.append(x_train[i])
  elif y_train[i] == 6:
    img_six.append(x_train[i])
  else:
    pass

img_zero = np.expand_dims(np.array(img_zero), axis=3)
img_one = np.expand_dims(np.array(img_one), axis=3)
img_two = np.expand_dims(np.array(img_two), axis=3)
img_three = np.expand_dims(np.array(img_three), axis=3)
img_six = np.expand_dims(np.array(img_six), axis=3)

label_zero = np.ones(len(img_zero), dtype=np.int32)*0
label_one = np.ones(len(img_one), dtype=np.int32)*1
label_two = np.ones(len(img_two), dtype=np.int32)*2
label_three = np.ones(len(img_three), dtype=np.int32)*3
label_six = np.ones(len(img_six), dtype=np.int32)*6

x_data = np.concatenate((img_zero, img_one, img_two), axis=0)
y_data = np.concatenate((label_zero, label_one, label_two), axis= 0)

print ('shape of x_data: ', x_data.shape)
print ('shape of y_data: ', y_data.shape)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, 
                                                 stratify=y_data,
                                                 random_state=10)
#sess.close()
tf.reset_default_graph()
#tf.compat.v1.reset_default_graph()

inputs = Input(shape=(28, 28, 1))
x = tfp.layers.Convolution2DFlipout(32, (5, 5), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = tfp.layers.Convolution2DFlipout(64, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = tfp.layers.DenseFlipout(128, activation='relu')(x)
x = Dropout(0.25)(x)
outputs = tfp.layers.DenseFlipout(3, activation=None)(x)

model = Model(inputs, outputs)

model.summary()
x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int32)
n = tf.placeholder(shape=[], dtype=tf.float32) 

logits = model(x)
probs = tf.nn.softmax(logits, axis=1)
labels_distribution = tfp.distributions.Categorical(logits=logits)
log_probs = labels_distribution.log_prob(y)

neg_log_likelihood = -tf.reduce_mean(log_probs)
kl = sum(model.losses) / n
elbo_loss = neg_log_likelihood + kl

correct_preds = tf.equal(tf.cast(y, dtype=tf.int64), tf.argmax(probs, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(elbo_loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batch_size = 32
n_iter = 100
show_step = 50

history_loss_train = []
history_acc_train = []
history_loss_val = []
history_acc_val = []

for i in range(n_iter):
  batch_indices = np.random.choice(len(x_train), batch_size, replace=False)
  batch_x = x_train[batch_indices] 
  batch_y = y_train[batch_indices]

  feed_dict = {x: batch_x, y: batch_y, n: batch_size}
  sess.run(train_op, feed_dict=feed_dict )

  temp_loss, temp_acc = sess.run([elbo_loss, accuracy], 
                                        feed_dict=feed_dict)

  history_loss_train.append(temp_loss)
  history_acc_train.append(temp_acc)

  if (i+1) % show_step == 0:
    print ('-' * 70)
    print ('Iteration: ' + str(i+1) + '  Loss: ' + str(temp_loss) 
           + '  Accuracy: ' + str(temp_acc))

  batch_indices = np.random.choice(len(x_val), batch_size, replace=False)
  batch_x = x_val[batch_indices] 
  batch_y = y_val[batch_indices]

  feed_dict = {x: batch_x, y: batch_y, n: batch_size}
  sess.run(train_op, feed_dict=feed_dict )

  temp_loss, temp_acc = sess.run([elbo_loss, accuracy], 
                                        feed_dict=feed_dict)

  history_loss_val.append(temp_loss)
  history_acc_val.append(temp_acc)
fig = plt.figure(figsize = (10, 3))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(n_iter), history_loss_train, 'b-', label='Training')
ax1.plot(range(n_iter), history_loss_val, 'r-', label='Validation')
ax1.set_title('Loss')
ax1.legend(loc='best')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(n_iter), history_acc_train, 'b-', label='Training')
ax2.plot(range(n_iter), history_acc_val, 'r-', label='Validation')
ax2.set_title('Accuracy')
ax2.legend(loc='best')

plt.show()   
idx = np.random.randint(1000)
n_samples = 100

#x_test = np.expand_dims(img_zero[idx], axis=0)
#y_true = 0

#x_test = np.expand_dims(img_two[idx], axis=0)
#y_true = 2

#x_test = np.expand_dims(img_three[idx], axis=0)
#y_true = 3

x_test = np.expand_dims(img_six[idx], axis=0)
y_true = 6

y_test_prob = []
feed_dict = {x: x_test, n: 1}
for i in range(n_samples):
  y_test_prob.append(sess.run(probs, feed_dict=feed_dict))

y_test_prob = np.squeeze(np.array(y_test_prob))
print ('Input: ', y_true)
print ()

plt.figure(figsize=(5, 3))
plt.imshow(np.squeeze(x_test), cmap='Greys')
plt.title('Image')
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.show()

fig = plt.figure(figsize = (10, 3))
ax1 = fig.add_subplot(1, 3, 1)
ax1.hist(y_test_prob[:, 0], range=(0.0, 1.0))
ax1.set_title('0, median: ' + str(np.median(y_test_prob[:, 0])))

ax2 = fig.add_subplot(1, 3, 2)
ax2.hist(y_test_prob[:, 1], range=(0.0, 1.0))
ax2.set_title('1, median: ' + str(np.median(y_test_prob[:, 1])))

ax3 = fig.add_subplot(1, 3, 3)
ax3.hist(y_test_prob[:, 2], range=(0.0, 1.0))
ax3.set_title('2, median: ' + str(np.median(y_test_prob[:, 2])))

plt.show()

