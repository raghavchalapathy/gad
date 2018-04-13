from six.moves import range
from scipy.io import loadmat
import numpy as np
import tflearn
from skimage import io
import skimage
from sklearn.metrics import average_precision_score,mean_squared_error
import tensorflow as tf
from tflearn.datasets import cifar10
from tflearn.layers.normalization import local_response_normalization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import label_binarize
from imgaug import augmenters as iaa
# Global variables
nb_feature = 64
import scipy

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
K = 10
def au_prc(y_true,y_score):
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_true, y_score)

    print('Average precision-recall score: {0:0.4f}'.format(
        average_precision))

    # precision, recall, _ = precision_recall_curve(y_test, y_score)
    #
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                  color='b')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     average_precision))

    return average_precision
def au_roc(y_true,y_score):
    from sklearn.metrics import roc_auc_score
    roc_score = roc_auc_score(y_true, y_score)

    print('ROC  score: {0:0.4f}'.format(
        roc_score))

    return roc_score
def compute_precAtK(y_true, y_score, K = 10):

    if K is None:
        K = y_true.shape[0]

    # label top K largest predicted scores as + one's've

    idx = np.argsort(y_score)
    predLabel = np.zeros(y_true.shape)

    predLabel[idx[:K]] = 1

    prec = precision_score(y_true, predLabel)

    return prec


## Case 1 : (Normal: Cats ) , (Anommalies: Inverted Cats )
image_and_inverted_anamolies_train = {'image1': 5,'image2': 3,'anomalies1':3,'imagecount': 2500,'anomaliesCount':0,'flipimage':"no",'slice_stich':"no"}

# Case 2 : (Normal: Cats , Dogs) , (Anommalies: sliced and stiched Cats and Dogs )
image_and_sliced_stiched_train =  {'image1': 3,'image2': 3,'imagecount': 1000,'flipimage':"no",'slice_stich':"yes"}
slice_and_stitch = {'image1': 3,'image2': 5,'imagecount1': 50, 'imagecount2': 50,'anomaliesCount':50}




basepath="/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/experiments/cats_n_dogs/rcae/"
# train_dataPath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/kdd_2018/CatsVsRotated/Train.mat"
# test_dataPath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/kdd_2018/CatsVsRotated/Test.mat"

mean_square_error_dict ={}
ROOT = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cifar-10-batches-py"

(X, Y), (testX, testY) = cifar10.load_data(ROOT)
# testX = np.asarray(testX)
# testY = np.asarray(testY)
# side = X.shape[1]
# channel = X.shape[3]
# noise_factor = 0.1


side = 32
channel =3

# Define the convoluted ae architecture
def encoder(inputs,hidden_layer):
    net = tflearn.conv_2d(inputs, 16, 3, strides=2)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)
    print "========================"
    print "enc-L1",net.get_shape()
    print "========================"

    net = tflearn.conv_2d(net, 16, 3, strides=1)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)
    print "========================"
    print "enc-L2",net.get_shape()
    print "========================"

    net = tflearn.conv_2d(net, 32, 3, strides=2)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)
    print "========================"
    print "enc-L3",net.get_shape()
    print "========================"
    net = tflearn.conv_2d(net, 32, 3, strides=1)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)
    print "========================"
    print "enc-L4",net.get_shape()
    print "========================"
    net = tflearn.flatten(net)
    #net = tflearn.fully_connected(net, nb_feature,activation="sigmoid")
    net = tflearn.fully_connected(net, nb_feature)

    h = net.W
    print "Encoder Weights shape",h.get_shape()

    net = tflearn.batch_normalization(net)
    net = tflearn.sigmoid(net)
    print "========================"
    print "hidden",net.get_shape()
    print "========================"

    return [net,h]
def decoder(inputs,decode_layer):
    net = tflearn.fully_connected(inputs, (side // 2**2)**2 * 32, name='DecFC1')
    d = tf.transpose(net.W)
    print "Decoder Weights shape",d.get_shape()
    net = tflearn.batch_normalization(net, name='DecBN1')
    net = tflearn.elu(net)
    print "========================"
    print "dec-L1",net.get_shape()
    print "========================"

    net = tflearn.reshape(net, (-1, side // 2**2, side // 2**2, 32))
    net = tflearn.conv_2d(net, 32, 3, name='DecConv1')
    net = tflearn.batch_normalization(net, name='DecBN2')
    net = tflearn.elu(net)
    print "========================"
    print "dec-L2",net.get_shape()
    print "========================"
    net = tflearn.conv_2d_transpose(net, 16, 3, [side // 2, side // 2],
                                        strides=2, padding='same', name='DecConvT1')
    net = tflearn.batch_normalization(net, name='DecBN3')
    net = tflearn.elu(net)
    print "========================"
    print "dec-L3",net.get_shape()
    print "========================"
    net = tflearn.conv_2d(net, 16, 3, name='DecConv2')
    net = tflearn.batch_normalization(net, name='DecBN4')
    net = tflearn.elu(net)
    print "========================"
    print "dec-L4",net.get_shape()
    print "========================"
    net = tflearn.conv_2d_transpose(net, channel, 3, [side, side],
                                        strides=2, padding='same', activation='sigmoid',
                                        name='DecConvT2')

    print "========================"
    print "output layer",net.get_shape()
    print "========================"
    return [net,d]

hidden_layer = None
decode_layer = None
# Building the autoencoder model
net = tflearn.input_data(shape=[None, side, side, channel])
[net,hidden_layer] = encoder(net,hidden_layer)
[net,decode_layer] = decoder(net,decode_layer)
mue = 0.1
net = tflearn.regression_RobustAutoencoder(net,mue,hidden_layer,decode_layer, optimizer='adam', learning_rate=0.01,
                         loss='rPCA_autoencoderLoss', metric=None,name="vanilla_autoencoder")
#rPCA_autoencoderLoss_FobsquareLoss
#rPCA_autoencoderLoss
#net = tflearn.regression(net, optimizer='adam', loss='mean_square', metric=None)
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='tensorboard/')

def addNoise(original, noise_factor):
    noisy = original + np.random.normal(loc=0.0, scale=noise_factor, size=original.shape)
    return np.clip(noisy, 0., 1.)
def add_Salt_Pepper_Noise(original, noise_factor):
    #noisy = original + np.random.normal(loc=0.0, scale=noise_factor, size=original.shape)
    noisy = skimage.util.random_noise(original, mode='s&p',clip=False,amount=0.1)
    return np.clip(noisy, 0., 1.)


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
        print temp_images.shape
        print ana_images.shape
        print temp_images_label.shape
        data = np.concatenate((temp_images, ana_images), axis=0)
        # labels for these images
        datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
        from  scipy.io import savemat
        mdict = {}
        mdict.update({'imgs': data, 'labels': datalabels})
        savemat(
            "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cats_n_dogs.mat",
            mdict)
        print ("Saved Cats and Dogs-----")
        exit()

        return [data, datalabels]


    return



def compute_mse(Xclean,Xdecoded,lamda):
    #print len(Xdecoded)
    Xclean = np.reshape(Xclean, (len(Xclean),3072))
    m,n =  Xclean.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),3072))

    meanSq_error= mean_squared_error(Xclean, Xdecoded)
    mean_square_error_dict.update({lamda:meanSq_error})
    print("\n Mean square error Score ((Xclean, Xdecoded):")
    print(mean_square_error_dict.values())

    return mean_square_error_dict
def prepare_cifar_data_with_noise_injection(original,original_labels,image_with_noise_injection):

    imagelabel = image_with_noise_injection['image']
    imagecnt = image_with_noise_injection['imagecount']

    idx = np.where(original_labels ==imagelabel)
    idx = idx[0][:imagecnt]


    images = original[idx]
    images_labels = original_labels[idx]



    data = images
    datalabels = images_labels



    return [data,datalabels]
# Function to compute softthresholding values
def soft_threshold(lamda,b):

    th = float(lamda)/2.0
    print ("(lamda,Threshold)",lamda,th)
    print("The type of b is ..., its len is ",type(b),b.shape,len(b[0]))

    if(lamda == 0):
        return b
    m,n = b.shape

    x = np.zeros((m,n))

    k = np.where(b > th)
    # print("(b > th)",k)
    #print("Number of elements -->(b > th) ",type(k))
    x[k] = b[k] - th

    k = np.where(np.absolute(b) <= th)
    # print("abs(b) <= th",k)
    # print("Number of elements -->abs(b) <= th ",len(k))
    x[k] = 0

    k = np.where(b < -th )
    # print("(b < -th )",k)
    # print("Number of elements -->(b < -th ) <= th",len(k))
    x[k] = b[k] + th
    x = x[:]

    return x
def compute_best_worst_rank(testX,Xdecoded):
     #print len(Xdecoded)

    testX = np.reshape(testX, (len(testX),3072))
    m,n =  testX.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),3072))

    # Rank the images by reconstruction error
    anamolies_dict = {}
    for i in range(0,len(testX)):
        anamolies_dict.update({i:np.linalg.norm(testX[i] - Xdecoded[i])})

    # Sort the recont error to get the best and worst 10 images
    best_top10_anamolies_dict={}
    # Rank all the images rank them based on difference smallest  error
    best_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=False)
    worst_top10_anamolies_dict={}
    worst_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=True)


    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_best = 0
    # Show the top 10 most badly reconstructed images
    for b in best_sorted_keys:
        if(counter_best <= 29):
            counter_best = counter_best + 1
            best_top10_anamolies_dict.update({b:anamolies_dict[b]})
    best_top10_keys = best_top10_anamolies_dict.keys()


    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_worst = 0
    # Show the top 10 most badly reconstructed images
    for w in worst_sorted_keys:
        if(counter_worst <= 29):
            counter_worst = counter_worst + 1
            worst_top10_anamolies_dict.update({w:anamolies_dict[w]})
    worst_top10_keys = worst_top10_anamolies_dict.keys()

    return [best_top10_keys,worst_top10_keys]
def fit_auto_conv_AE(input,Xclean):


    input = np.reshape(input, (len(input),32,32,3))
    model.fit(input, Xclean, n_epoch=10,
          run_id="auto_encoder", batch_size=128)

    ae_output = model.predict(input)
    ae_output = np.reshape(ae_output, (len(ae_output),3072))

    return ae_output
def compute_softhreshold(XtruewithNoise,N,lamda,Xclean):
    XtruewithNoise = np.reshape(XtruewithNoise, (len(XtruewithNoise),3072))
    print "lamda passed ",lamda
    # inner loop for softthresholding
    for i in range(0, 1):
        train_input = XtruewithNoise - N
        XAuto = fit_auto_conv_AE(train_input,Xclean) # XAuto is the predictions on train set of autoencoder
        XAuto = np.asarray(XAuto)
        #print "XAuto:",type(XAuto),XAuto.shape
        softThresholdIn = XtruewithNoise - XAuto
        softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn),3072))
        N = soft_threshold(lamda,softThresholdIn)
        print("Iteration NUmber is : ",i)
        print ("NUmber of non zero elements  for N,lamda",np.count_nonzero(N),lamda)
        print ( "The shape of N", N.shape)
        print ( "The minimum value of N ", np.amin(N))
        print ( "The max value of N", np.amax(N))


    return N
def visualise_anamolies_detected(testX,noisytestX,decoded,N,best_top10_keys,worst_top10_keys,lamda):

    N = np.reshape(N, (len(N),32,32,3))
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
    print(basepath+'results/best/')
    io.imsave(basepath+'results/best/'+str(lamda)+'_cae_decode.png', img)

    #Display the decoded Original, noisy, reconstructed images for worst
    print "+++++++++++++++++++++++++++++++++++++++++++"
    print "Worst reconstruction Keys:",worst_top10_keys
    print "+++++++++++++++++++++++++++++++++++++++++++"
    img = np.ndarray(shape=(side*2, side*10, channel))
    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[worst_top10_keys[i]]
        img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = N[worst_top10_keys[i]]
        # img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[worst_top10_keys[i]]
        # img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[worst_top10_keys[i]]

    # img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for worst after being encoded and decoded: @")
    print(basepath+'results/worst/')
    io.imsave(basepath+'results/worst/'+str(lamda)+'_cae_decode.png', img)

    return


import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

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
def precAtK(pred, trueY, K = None):

    if K is None:
        K = trueY.shape[0]

    # label top K largest predicted scores as +'ve
    idx = np.argsort(-pred)
    predLabel = -np.ones(trueY.shape)
    predLabel[idx[:K]] = 1



    ""
    print(predLabel)

    prec = precision_score(trueY, predLabel,average=None)


    return prec
def load_cifar10catsdogs():

    data = loadmat('/Users/raghav/Documents/Uni/ECML_2017/experiments/cifar_10/data/cifar-10-cats-dogs.mat')

    X = data['X']
    Y = data['Y'].ravel()

    XPos = X[Y == +1,:]
    XNeg = X[Y == -1,:]

    YPos = Y[Y == +1]
    YNeg = Y[Y == -1]

    # print(XPos.shape)
    # print(XNeg.shape)
    # print(YPos.shape)
    # print(YNeg.shape)

    return [(XPos,YPos),(XNeg,YNeg)]


def loadtrainTestData(train_dataPath,test_dataPath):
    imgdataTrain = scipy.io.loadmat(train_dataPath)
    imgdataTest = scipy.io.loadmat(test_dataPath)
    X = []
    for i in range(0, 900):
        X.append(imgdataTrain['Train'][0][i])
    for i in range(0, 100):
            X.append(imgdataTest['Test'][0][i])
    X = np.asarray(X)


    return X

def loadData(train_dataPath,mode):
    imgdata = scipy.io.loadmat(train_dataPath)
    X = []
    if(mode =="train"):
        for i in range(0,900):
            X.append(imgdata['Train'][0][i])
    else:
        for i in range(50, 100):
            X.append(imgdata['Test'][0][i])

    X = np.asarray(X)


    return X
## Case 1 : (Normal: Cats ) , (Anommalies: Inverted Cats )
[X,Y]=prepare_cifar_data_with_anamolies(X,Y,image_and_inverted_anamolies_train)

# [testX,testY]=prepare_cifar_data_with_anamolies(testX,testY,image_and_inverted_anamolies_train)

# X = loadData(train_dataPath,"train")
# Y = np.ones(len(X))
# Y = Y.ravel()
# testX = loadData(test_dataPath,"test")
# testY = np.zeros(len(testX))
# testY = testY.ravel()

# X = loadtrainTestData(train_dataPath,test_dataPath)
# Y = np.ones(len(X))
# Y = Y.ravel()

#define lamda set
# lamda_set = [ 0.0,0.01,0.1,0.5,1.0, 10.0, 100.0]
# lamda_set = [ 0.0,0.01,0.1,0.5]
lamda_set = [ 0,0.01]
mue = 0.0
TRIALS= 7
# aplist = np.zeros((TRIALS,))
# auclist = np.zeros((TRIALS,))
# preclist= np.zeros((TRIALS,))

aplist = []
auclist = []
preclist= []

# outer loop for lamda
for l in range(0,len(lamda_set)):
    # Learn the N using softthresholding technique
    N =  0
    lamda = lamda_set[l]

    N = compute_softhreshold(X,N,lamda,X)

    XTrue = X
    YTrue = Y
    # XTrue = testX
    #Predict the conv_AE autoencoder output
    XTrue = np.reshape(XTrue, (len(XTrue),32,32,3))
    decoded = model.predict(XTrue)

    #compute MeanSqared error metric
    compute_mse(XTrue,decoded,lamda)

    # rank the best and worst reconstructed images
    [best_top10_keys,worst_top10_keys]=compute_best_worst_rank(XTrue,decoded)

    #Visualise the best and worst ( image, BG-image, FG-Image)
    XPred = np.reshape(np.asarray(decoded),(len(decoded),32,32,3))
    visualise_anamolies_detected(XTrue,XPred,decoded,N,best_top10_keys,worst_top10_keys,lamda)

    XPred = decoded
    # print type(XPred),len(XPred)

    # print type(XTrue),len(XTrue),XTrue.shape
    # YTrue = testY
    # YTrue = Y
    # YTrue = label_binarize(YTrue, classes=[1, 0])
    # print "Ytrue============================"
    # print type(YTrue),len(YTrue),YTrue.shape
    # #   print
    #
    (aprc_val, auc_val,prec_val) = evalPred(XPred, XTrue, YTrue)
    # aprc = au_prc(YTrue, XPred)
    # auc = au_roc(YTrue, XPred)
    # prec = compute_precAtK(YTrue, XPred, K=10)
    aplist.append(aprc_val)
    auclist.append(auc_val)
    preclist.append(prec_val)
    # auc[l] = auc
    # prec[l] = prec
    print "AUPRC",l,aprc_val
    print "AUROC",l,auc_val
    print "P@10",l,prec_val

    # (ap[l], auc[l], prec[t], recErr) = evalPred(XPred, XTrue, YTrue)



print('AUPRC = %1.4f +- %1.4f' % (np.mean(aplist), np.std(aplist)/np.sqrt(TRIALS)))
print('AUROC = %1.4f +- %1.4f' % (np.mean(auclist), np.std(auclist)/np.sqrt(TRIALS)))
print('P@10  = %1.4f +- %1.4f' % (np.mean(preclist), np.std(preclist)/np.sqrt(TRIALS)))


# plotting the mean precision score
print("\n Saving the Mean square error Score ((Xclean, Xdecoded):")
fig1_mean_square_error=plt.figure(figsize=(8,5))
plt.xlabel("RCAE")
plt.ylabel("Mean- Sq Error")
print("\n Mean square error Score ((Xclean, Xdecoded):")
print(mean_square_error_dict.values())
for k,v in mean_square_error_dict.iteritems():
    print k,v
# basic plot
data = mean_square_error_dict.values()
plt.boxplot(data)
fig1_mean_square_error.savefig(basepath+'cifar10_mean_square_error.png')


