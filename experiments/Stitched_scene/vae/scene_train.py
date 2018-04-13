import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cPickle

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from tflearn.datasets import cifar10
from imgaug import augmenters as iaa
# import parameters
from cifar10_params import *

# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

# mean and variance for latent variables
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# decoder architecture
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * img_rows / 2 * img_cols / 2, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, img_rows / 2, img_cols / 2)
else:
    output_shape = (batch_size, img_rows / 2, img_cols / 2, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


# load dataset
# (x_train, _), (x_test, y_test) = cifar10.load_data()
# x_test = x_test.astype('float32') / 255.
# x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
image_and_inverted_anamolies_train = {'image1': 5,'image2': 3,'anomalies1':3,'imagecount': 2500,'anomaliesCount':0,'flipimage':"no",'slice_stich':"no"}
# Case 2 : (Normal: Cats , Dogs) , (Anommalies: sliced and stiched Cats and Dogs )
image_and_sliced_stiched_train =  {'image1': 3,'image2': 3,'imagecount': 1000,'flipimage':"no",'slice_stich':"yes"}
slice_and_stitch = {'image1': 3,'image2': 5,'imagecount1': 50, 'imagecount2': 50,'anomaliesCount':50}
def prepare_cifar_data_with_anamolies(original,original_labels,image_and_anamolies):

    image1_label = image_and_anamolies['image1']
    image2_label = image_and_anamolies['image2']

    imagecnt = image_and_anamolies['imagecount']
    flipImage = image_and_anamolies['flipimage']
    slice_stich = image_and_anamolies['slice_stich']

    import numpy as np
    idx1 = np.where(original_labels ==image1_label)
    idx2 = np.where(original_labels == image2_label)
    idx3 = np.where(original_labels == image2_label)

    anamoliescnt = image_and_anamolies['anomaliesCount']
    anamolieslabel1 = image_and_anamolies['anomalies1']

    idx1 = idx1[0][:imagecnt]
    idx2 = idx2[0][:imagecnt]
    idx3 = idx3[0][:anamoliescnt]


    images1 = original[idx1]
    images2 = original[idx2]
    images3 = original[idx3]

    images_labels1 = original_labels[idx1]
    images_labels2 = original_labels[idx2]


    temp_images = np.concatenate((images1, images2), axis=0)
    temp_images_label = np.concatenate((images_labels1, images_labels2), axis=0)

    temp_images_label = np.ones(len(temp_images)) ## Assign label 1 for normal cats and 0 for rotated cats

    anomalies_images = images3
    anomalies_labels = original_labels[idx3]




    # if(anamoliescnt == 0):
    #     seq = iaa.Sequential([
    #            iaa.Flipud(0.9)  # vertically flip 90% of the images
    #     ])
    #     anoidx = np.where(original_labels == anamolieslabel1)
    #     anomaliesidx1 = anoidx[0][:50]
    #     anomalies_img = original[anomaliesidx1]
    #     anomalies_labels = original_labels[anomaliesidx1]
    #     ana_images = anomalies_img
    #     ana_images *= 255
    #     ana_images = ana_images.astype(np.uint8)
    #     ana_images = seq.augment_images(ana_images)
    #     ana_labels = np.full(len(ana_images), 3)
    #     data = np.concatenate((temp_images, ana_images), axis=0)
    #     # labels for these images
    #     datalabels = np.concatenate((temp_images_label, anomalies_labels), axis=0)
    #     return [data, datalabels]


    if(flipImage == "yes"):
        seq = iaa.Sequential([
               iaa.Flipud(1.0)  # vertically flip 90% of the images
        ])
        ana_images = anomalies_images
        ana_images *= 255
        ana_images = ana_images.astype(np.uint8)
        ana_images = seq.augment_images(ana_images)
        ana_labels = np.full(len(ana_images), -1)
        data = np.concatenate((temp_images, ana_images), axis=0)
        # labels for these images
        datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
        from  scipy.io import savemat
        mdict = {}
        mdict.update({'imgs':data,'labels':datalabels})
        savemat("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cats_n_invertedCats.mat", mdict)

        return [data, datalabels]

    if(slice_stich == "yes"):

           [ana_images,ana_labels] = func_slice_stich(original, original_labels)
           data = np.concatenate((temp_images, ana_images), axis=0)
           # labels for these images
           print len(temp_images_label),len(ana_labels)
           datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
           return [data, datalabels]

    else:
        # Read the anomalies from file to numpy array and assign the labels for anomalies as 99
        from PIL import Image
        import os, numpy as np
        folder = '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/kdd_2018/dogs_n_dogs_with_cat/cats_n_dogs/'
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        ims = [read(os.path.join(folder, filename)) for filename in os.listdir(folder)]
        ana_images = np.array(ims)
        ana_labels = np.full(len(ana_images), 3)
        # print temp_images.shape
        # print ana_images.shape
        # print temp_images_label.shape
        data = np.concatenate((temp_images, ana_images), axis=0)
        # labels for these images
        datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
        from  scipy.io import savemat
        mdict = {}
        mdict.update({'imgs': data, 'labels': datalabels})
        savemat("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cats_n_dogs.mat",
            mdict)

        return [data, datalabels]


    return
def prepare_scene_data_with_anomalies(normalPath, anomaliesPath):

    # Read the anomalies from file to numpy array and assign the labels for anomalies as 99
    train_images = readjpegimages2Array(normalPath)
    ana_images = readjpegimages2Array(anomaliesPath)

    print train_images.shape

    train_labels = np.full(len(train_images), 1)
    ana_labels   = np.full(len(ana_images), -1)  # give anomalies label one among the normal dataset

    data = np.concatenate((train_images, ana_images), axis=0)
    data = data/255.0 ## normalise the data
    # labels for these images
    datalabels = np.concatenate((train_labels, ana_labels), axis=0)
    return [data, datalabels]

    return
def func_slice_stich(X_images,Y_labels):

    idx1 = np.where(Y_labels == slice_and_stitch['image1'])
    idx2 = np.where(Y_labels == slice_and_stitch['image2'])

    imagecnt = int(slice_and_stitch['imagecount2'])
    print "imageCount", imagecnt



    idx1 = idx1[0][:imagecnt]
    idx2 = idx2[0][:imagecnt]

    images1 = X_images[idx1]
    images2 = X_images[idx2]

    images_labels1 = Y_labels[idx1]
    images_labels2 = Y_labels[idx2]

    # data_labels = np.concatenate((images_labels1, images_labels2), axis=0)

    res1 = np.array_split(images1, 2, axis=1)
    res2 = np.array_split(images2, 2, axis=1)
    data = np.hstack((res1[0], res2[0]))
    reslabel1 = np.array_split(images_labels1, 2)
    reslabel2 = np.array_split(images_labels1, 2)
    data_labels = np.hstack((reslabel1[0], reslabel2[0]))

    # imgs =  np.concatenate((images1, images2), axis=0)

    return [data,data_labels]
ROOT = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cifar-10-batches-py"
normal_dataPath ="/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/kdd_2018/StitchedData/normal"
anomalies_dataPath="/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/kdd_2018/StitchedData/anomalies/"
def readjpegimages2Array(filepath):
    from PIL import Image
    import os, numpy as np
    import matplotlib.pyplot as plt
    folder = filepath
    read = lambda imname: np.asarray(Image.open(imname))
    ims = [np.array(Image.open(os.path.join(folder, filename)).convert('RGB')) for filename in os.listdir(folder)]
    imageList = []
    for x in range(0,len(ims)):
        # print "ims[x].shape",ims[x].shape
        if(ims[x].shape ==(32,32,3)):
            imageList.append(ims[x])
    result = np.asarray(imageList)

    return result
[x_train,Y] = prepare_scene_data_with_anomalies(normal_dataPath, anomalies_dataPath)
# x_train = X.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)

basepath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/experiments/Stitched_scene/vae/"

# training
history = vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_train, None))


# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


#Compute the difference between actual and the generated image and sort the errors by decreasing order.


# save all 3 models for future use - especially generator
vae.save(basepath+'/models/scene_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
encoder.save(basepath+'/models/scene_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
generator.save(basepath+'/models/scene_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs))

# save training history
fname = basepath+'/models/scene_ld_%d_conv_%d_id_%d_e_%d_history.pkl' % (latent_dim, num_conv, intermediate_dim, epochs)
with open(fname, 'wb') as file_pi:
    cPickle.dump(history.history, file_pi)
