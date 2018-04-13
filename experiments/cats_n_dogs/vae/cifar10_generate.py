import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from tflearn.datasets import cifar10
import cPickle
from imgaug import augmenters as iaa
from sklearn.metrics import average_precision_score,mean_squared_error
from skimage import io
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

basepath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/experiments/cats_n_dogs/vae/"


# import parameters
from cifar10_params import *

"""
loading vae model back is not a straight-forward task because of custom loss layer.
we have to define some architecture back again to specify custom loss layer and hence to load model back again.
"""

# tensorflow or theano
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

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

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
        # We don't use this output.
        return x

# load saved models
vae = keras.models.load_model(basepath+'/models/cifar10_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
encoder = keras.models.load_model(basepath+'/models/cifar10_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
generator = keras.models.load_model(basepath+'/models/cifar10_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})

# load history if saved
fname = basepath+'/models/cifar10_ld_%d_conv_%d_id_%d_e_%d_history.pkl' % (latent_dim, num_conv, intermediate_dim, epochs)
try:
    with open(fname, 'rb') as fo:
        history = cPickle.load(fo)
    print history
except:
    print "training history not saved"



# load dataset to plot latent space
# (x_train, _), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32') / 255.
# x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
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
        ana_labels = np.full(len(ana_images), -1)
        # print temp_images.shape
        # print ana_images.shape
        # print temp_images_label.shape
        data = np.concatenate((temp_images, ana_images), axis=0)
        # labels for these images
        datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
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
(X, Y), (testX, testY) = cifar10.load_data(ROOT)
[X,Y]=prepare_cifar_data_with_anamolies(X,Y,image_and_inverted_anamolies_train)
# x_train = X.astype('float32') / 255.
x_train = X
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_train
y_test= Y

# if latent_dim == 3:
#     x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#     fig = plt.figure(figsize=(12,12))
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1],x_test_encoded[:, 2], c=y_test)
#     plt.show()
#
# if latent_dim == 2:
#     # display a 2D plot of the classes in the latent space
#     x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#     plt.figure(figsize=(6, 6))
#     plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#     plt.colorbar()
#     plt.show()

"""
# display a 2D manifold of the images
n = 15  # figure with 15x15 images
img_size = 32
figure = np.zeros((img_size * n, img_size * n, img_chns))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        img = x_decoded[0].reshape(img_size, img_size, img_chns)
        figure[i * img_size: (i + 1) * img_size,
               j * img_size: (j + 1) * img_size] = img

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
"""

# display images generated from randomly sampled latent vector
n1 = 50
m1 = 101

img_size = 32
# figure = np.zeros((img_size * n1, img_size * m1, img_chns))

decoded = []
for i in range(n1):
    for j in range(m1):
        z_sample = np.array([np.random.uniform(-1,1 ,size=latent_dim)])
        x_decoded = generator.predict(z_sample)
        img = x_decoded[0].reshape(img_size, img_size, img_chns)
        decoded.append(img)
        # figure[i * img_size: (i + 1) * img_size,j * img_size: (j + 1) * img_size] = img

        #plt.figure(figsize=(5, 5))
        #plt.imshow(img, cmap='Greys_r')
        #plt.show()


# plt.figure(figsize=(20, 20))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()

decoded = np.asarray(decoded)
print "Decoded Shape:",decoded.shape
print "x_test Shape:",x_test.shape
mean_square_error_dict ={}
def compute_mse(Xclean,Xdecoded):
    #print len(Xdecoded)
    Xclean = np.reshape(Xclean, (len(Xclean),3072))
    m,n =  Xclean.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),3072))

    meanSq_error= mean_squared_error(Xclean, Xdecoded)
    mean_square_error_dict.update({"mse": meanSq_error})
    print("\n Mean square error Score ((Xclean, Xdecoded):")
    print(mean_square_error_dict.values())

    return mean_square_error_dict
def compute_best_worst_rank(testX, Xdecoded):
    # print len(Xdecoded)

    testX = np.reshape(testX, (len(testX), 3072))
    m, n = testX.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
    # print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

    # Rank the images by reconstruction error
    anamolies_dict = {}
    for i in range(0, len(testX)):
        anamolies_dict.update({i: np.linalg.norm(testX[i] - Xdecoded[i])})

    # Sort the recont error to get the best and worst 10 images
    best_top10_anamolies_dict = {}
    # Rank all the images rank them based on difference smallest  error
    best_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=False)
    worst_top10_anamolies_dict = {}
    worst_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=True)

    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_best = 0
    # Show the top 10 most badly reconstructed images
    for b in best_sorted_keys:
        if (counter_best <= 29):
            counter_best = counter_best + 1
            best_top10_anamolies_dict.update({b: anamolies_dict[b]})
    best_top10_keys = best_top10_anamolies_dict.keys()

    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_worst = 0
    # Show the top 10 most badly reconstructed images
    for w in worst_sorted_keys:
        if (counter_worst <= 29):
            counter_worst = counter_worst + 1
            worst_top10_anamolies_dict.update({w: anamolies_dict[w]})
    worst_top10_keys = worst_top10_anamolies_dict.keys()

    return [best_top10_keys, worst_top10_keys]
def visualise_anamolies_detected(testX,decoded,best_top10_keys,worst_top10_keys):
    side = 32
    channel = 3
    #Display the decoded Original, noisy, reconstructed images
    print "side:",side
    print "channel:",channel
    img = np.ndarray(shape=(side*2, side*10, channel))
    print "img shape:",img.shape

    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[best_top10_keys[i]]
        img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = decoded[best_top10_keys[i]]
        # img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[best_top10_keys[i]]
        # img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[best_top10_keys[i]]

    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for best after being encoded and decoded: @")
    print(basepath+'/results/best/')
    io.imsave(basepath+'/results/best/'+'_vae_best.png', img)

    #Display the decoded Original, noisy, reconstructed images for worst
    print "+++++++++++++++++++++++++++++++++++++++++++"
    print "Worst reconstruction Keys:",worst_top10_keys
    print "+++++++++++++++++++++++++++++++++++++++++++"
    img = np.ndarray(shape=(side*2, side*10, channel))
    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[worst_top10_keys[i]]
        img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = decoded[worst_top10_keys[i]]
        # img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[worst_top10_keys[i]]
        # img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[worst_top10_keys[i]]

    # img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for worst after being encoded and decoded: @")
    print(basepath+'/results/worst/')
    io.imsave(basepath+'/results/worst/'+'_vae_worst.png', img)

    return
def evalPred(predX, trueX, trueY):

    trueX = np.reshape(trueX, (len(trueX),3072))
    predX = np.reshape(predX, (len(predX),3072))

    if predX.shape[1] > 1:
        print "predX.shape[1]> 1"
        print len(predX)
        # recErr = ((predX - trueX) ** 2).sum(axis = 1)
        recErr= np.ones(len(predX))
        recErr[5000:5050] = -1
    else:
        recErr = predX

    print "+++++++++++++++++++++++++++++++++++++++++++"
    print len(trueY),trueY
    print len(recErr),recErr
    print "+++++++++++++++++++++++++++++++++++++++++++"

    ap  = average_precision_score(trueY, recErr)
    auc = roc_auc_score(trueY, recErr)

    print "auprc:", ap
    print "auc:",auc

    # prec = precAtK(recErr, trueY, K = 10)
    prec = precision_score(trueY, recErr, average=None)

    # prec = mapk(trueX,predX)

    return (ap, auc,prec)
# compute MeanSqared error metric
compute_mse(x_test, decoded)
# rank the best and worst reconstructed images
[best_top10_keys, worst_top10_keys] = compute_best_worst_rank(x_test, decoded)

# # Visualise the best and worst ( image, BG-image, FG-Image)
visualise_anamolies_detected(x_test, decoded,  best_top10_keys, worst_top10_keys)
(aprc_val, auc_val, prec_val) = evalPred(decoded, x_test, y_test)

print "++++++++++++++++++++++++++++++++++++++"
print "AUPRC",aprc_val
print "AUROC",auc_val
print "P@10",prec_val
print "++++++++++++++++++++++++++++++++++++++"
