import platform, os
import urllib.request as img_down
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image

print("OS : " + platform.platform())

X = tf.placeholder(tf.float32, shape=[None, 150, 150, 3])

keep_prob = tf.placeholder(tf.float32)

training = tf.placeholder(tf.bool)

# Convolutional Layers
conv_0 = tf.layers.conv2d(X, filters=64, kernel_size=2, strides=2, padding='SAME', activation=tf.nn.relu)
batch_norm_0 = tf.layers.batch_normalization(conv_0, training=training)
max_pool_0 = tf.nn.max_pool(batch_norm_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

dropout_0 = tf.nn.dropout(max_pool_0, keep_prob)

conv_1 = tf.layers.conv2d(dropout_0, filters=256, kernel_size=2, strides=2, padding='SAME', activation=tf.nn.relu)
batch_norm_1 = tf.layers.batch_normalization(conv_1, training=training)
max_pool_1 = tf.nn.max_pool(batch_norm_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

dropout_1 = tf.nn.dropout(max_pool_1, keep_prob)

conv_2 = tf.layers.conv2d(dropout_1, filters=512, kernel_size=2, strides=2, padding='SAME', activation=tf.nn.relu)
batch_norm_2 = tf.layers.batch_normalization(conv_2, training=training)
max_pool_2 = tf.nn.max_pool(batch_norm_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Flatten & Output FCL
flatten_layer = tf.reshape(max_pool_2, [-1, 3 * 3 * 512])

dense_0 = tf.layers.dense(flatten_layer, 1500, activation=tf.nn.relu)
batch_norm_4 = tf.layers.batch_normalization(dense_0, training=training)

dense_1 = tf.layers.dense(batch_norm_4, 256, activation=tf.nn.relu)
batch_norm_5 = tf.layers.batch_normalization(dense_1, training=training)

dense_2 = tf.layers.dense(batch_norm_5, 64, activation=tf.nn.relu)
output_layer = tf.layers.dense(dense_2, 2)

sess = tf.Session()

saver = tf.train.Saver()

save_path = os.getcwd() + "/Server/utils/checkpoints/toto_banner_cnn.ckpt"

saver.restore(sess, save_path)


def check_os_type():
    if platform.system() == "Darwin":
        return "mac"

    elif platform.system() == "Linux":
        return "linux"


def check_toto_banner(array):
    paths = download_img(array)

    result = []

    for path in paths:
        result.append(model(path))

    return result


def download_img(array):
    image_path = []

    for image_link in array:
        randomString = str(uuid.uuid4()).replace("-", "")

        file_type = image_link.split(".")

        path = os.getcwd() + "/Server/image/" + randomString + "." + file_type[-1]

        img_down.urlretrieve(image_link, path)

        image_path.append(path)

    return image_path


def model(path):

    img = Image.open(path).convert('RGB')
    new_img = img.resize((150, 150))
    img_array = np.array(new_img)

    result = sess.run(output_layer, feed_dict={X: [img_array], keep_prob: 1.0, training: False})

    print(result)

    if (result[0][0] > result[0][1]):
        return True

    else:
        return False



