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


nb_feature = 64


image_and_anamolies_train = {'coast': 0,'insidecity': 1,'mountain': 2,'anomalies1':99,'imagecount': 80,'anomaliesCount':0,'slice_stich':"yes"}


image_and_anamolies_test = {'coast': 0,'insidecity': 1,'mountain':2,'anomalies1':99,'imagecount': 20,'anomaliesCount':20}
# image_with_noise_injection = {'image': 5,'imagecount': 5000}


basepath="/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/experiments/Stitched_scene/rcae/"
mean_square_error_dict ={}

ROOT = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cifar-10-batches-py"
side = 256
channel = 1
noise_factor = 0.1
img_dim = 256
dim = 65536



# Define the convoluted ae architecture
def encoder(inputs,hidden_layer):
    nb_feature=64
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
net = tflearn.regression_RobustAutoencoder(net,mue,hidden_layer,decode_layer, optimizer='adam', learning_rate=0.001,
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

def readjpegimages2Array(filepath):
    from PIL import Image
    import os, numpy as np
    import matplotlib.pyplot as plt
    folder = filepath
    read = lambda imname: np.asarray(Image.open(imname))
    ims = [np.array(Image.open(os.path.join(folder, filename)).convert('L')) for filename in os.listdir(folder)]
    imageList = []
    for x in range(0,len(ims)):

        if(ims[x].shape ==(256,256)):
            imageList.append(ims[x])
    result = np.asarray(imageList)

    return result

def func_slice_stich_scene(X1_images,X2_images,Y1_labels,Y2_labels):


    # data_labels = np.concatenate((images_labels1, images_labels2), axis=0)

    res1 = np.array_split(X1_images, 2, axis=1)
    res2 = np.array_split(X2_images, 2, axis=1)
    data = np.hstack((res1[0], res2[0]))
    reslabel1 = np.array_split(Y1_labels, 2)
    reslabel2 = np.array_split(Y2_labels, 2)
    data_labels = np.hstack((reslabel1[0], reslabel2[0]))

    # imgs =  np.concatenate((images1, images2), axis=0)

    return [data,data_labels]

def prepare_cifar_data_with_anamolies(image_and_anamolies):

    image1_label = image_and_anamolies['coast']
    image2_label = image_and_anamolies['insidecity']
    image3_label = image_and_anamolies['mountain']
    slice_stich = image_and_anamolies['slice_stich']

    imagecnt = image_and_anamolies['imagecount']

    # images1 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/train/coast/")
    # images2 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/train/insidecity/")
    # images3 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/train/mountains/")

    images1 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/train/coast/")
    images3 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/train/insidecity/")

    print images1.shape
    # print images2.shape
    print images3.shape
    import numpy as np
    temp_images = np.concatenate((images1,images3), axis=0)

    images_labels1 =  np.full(len(images1), 0)
    # images_labels2 = np.full(len(images2), 1)
    images_labels3 = np.full(len(images3), 1)
    temp_images_label = np.concatenate((images_labels1,images_labels3), axis=0)

    anamoliescnt = image_and_anamolies['anomaliesCount']
    anamolieslabel1 = image_and_anamolies['anomalies1']

    if(anamoliescnt == 0):
        return [temp_images, temp_images_label] # No anomalies during the training

    if(slice_stich == "yes"):

           [ana_images,ana_labels] = func_slice_stich_scene(images1,images3, images_labels1,images_labels3)
           data = np.concatenate((temp_images, ana_images), axis=0)
           # labels for these images
           print len(temp_images_label),len(ana_labels)
           datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
           return [data, datalabels]

    else:
        # Read the anomalies from file to numpy array and assign the labels for anomalies as 99
        test_images1 = readjpegimages2Array("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/test/")
        ana_images = readjpegimages2Array('/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/scene/grp2anomalies/')

        testimg_labels = np.full(len(test_images1), 0)
        ana_labels = np.full(len(ana_images), 1) # give anomalies label one among the normal dataset


        data = np.concatenate((test_images1, ana_images), axis=0)
        # labels for these images
        datalabels = np.concatenate((testimg_labels, ana_labels), axis=0)
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
    # labels for these images
    datalabels = np.concatenate((train_labels, ana_labels), axis=0)
    return [data, datalabels]

    return

def compute_mse(Xclean,Xdecoded,lamda):
    #print len(Xdecoded)
    Xclean = np.reshape(Xclean, (len(Xclean),dim))
    m,n =  Xclean.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),dim))

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

    testX = np.reshape(testX, (len(testX),dim))
    m,n =  testX.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),dim))

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


    input = np.reshape(input, (len(input),side,side,channel))
    print "++++++++++++++"
    print input.shape
    print Xclean.shape
    print "++++++++++++++"
    model.fit(input, Xclean, n_epoch=10,
          run_id="auto_encoder", batch_size=128)

    ae_output = model.predict(input)
    ae_output = np.reshape(ae_output, (len(ae_output),dim))

    return ae_output


def compute_softhreshold(XtruewithNoise,N,lamda,Xclean):
    XtruewithNoise = np.reshape(XtruewithNoise, (len(XtruewithNoise),dim))
    print "lamda passed ",lamda
    # inner loop for softthresholding
    for i in range(0, 1):
        train_input = XtruewithNoise - N
        XAuto = fit_auto_conv_AE(train_input,Xclean) # XAuto is the predictions on train set of autoencoder
        XAuto = np.asarray(XAuto)
        #print "XAuto:",type(XAuto),XAuto.shape
        softThresholdIn = XtruewithNoise - XAuto
        softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn),dim))
        N = soft_threshold(lamda,softThresholdIn)
        print("Iteration NUmber is : ",i)
        print ("NUmber of non zero elements  for N,lamda",np.count_nonzero(N),lamda)
        print ( "The shape of N", N.shape)
        print ( "The minimum value of N ", np.amin(N))
        print ( "The max value of N", np.amax(N))


    return N

def visualise_anamolies_detected(testX,noisytestX,decoded,N,best_top10_keys,worst_top10_keys,lamda):


    #Display the decoded Original, noisy, reconstructed images

    img = np.ndarray(shape=(side*3, side*10))
    print "img shape:",img.shape

    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1)] = np.transpose(np.reshape(testX[best_top10_keys[i]],(img_dim, img_dim)))
        img[side*(row+1):side*(row+2), side*col:side*(col+1)] = np.reshape(np.asarray(decoded[best_top10_keys[i]]),(img_dim, img_dim))
        img[side*(row+2):side*(row+3), side*col:side*(col+1)] = np.reshape(N[best_top10_keys[i]],(img_dim, img_dim))


    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for best after being encoded and decoded: @")
    print(basepath+'/results/best/')
    io.imsave(basepath+'/results/best/'+str(lamda)+'rcae_decode.png', img)

    #Display the decoded Original, noisy, reconstructed images for worst
    img = np.ndarray(shape=(side*3, side*10))
    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1)] = np.reshape(testX[worst_top10_keys[i]],(img_dim, img_dim))
        img[side*(row+1):side*(row+2), side*col:side*(col+1)] = np.reshape(np.asarray(decoded[worst_top10_keys[i]]),(img_dim, img_dim))
        img[side*(row+2):side*(row+3), side*col:side*(col+1)] = np.reshape(N[worst_top10_keys[i]],(img_dim, img_dim))


    img *= 255
    img = img.astype(np.uint8)

    # Save the image decoded
    print("\nSaving results for worst after being encoded and decoded: @")
    print(basepath+'/results/worst/')
    io.imsave(basepath+'/results/worst/'+str(lamda)+'rcae_decode.png', img)


    return
#
# def visualise_anamolies_detected(testX,noisytestX,decoded,N,best_top10_keys,worst_top10_keys,lamda):
#
#     N = np.reshape(N, (len(N),side,side,1))
#     #Display the decoded Original, noisy, reconstructed images
#     print "side:",side
#     print "channel:",channel
#     img = np.ndarray(shape=(side*4, side*10))
#     print "img shape:",img.shape
#
#     for i in range(10):
#         row = i // 10 * 3
#         col = i % 10
#         img[side*row:side*(row+1), side*col:side*(col+1)] = testX[best_top10_keys[i]]
#         img[side*(row+1):side*(row+2), side*col:side*(col+1)] = noisytestX[best_top10_keys[i]]
#         img[side*(row+2):side*(row+3), side*col:side*(col+1)] = decoded[best_top10_keys[i]]
#         img[side*(row+3):side*(row+4), side*col:side*(col+1)] = N[best_top10_keys[i]]
#
#     img *= 255
#     img = img.astype(np.uint8)
#
#     #Save the image decoded
#     print("\nSaving results for best after being encoded and decoded: @")
#     print(basepath+'/best/')
#     io.imsave(basepath+'/best/'+str(lamda)+'_cae_decode.png', img)
#
#     #Display the decoded Original, noisy, reconstructed images for worst
#     img = np.ndarray(shape=(side*4, side*10, channel))
#     for i in range(10):
#         row = i // 10 * 3
#         col = i % 10
#         img[side*row:side*(row+1), side*col:side*(col+1)] = testX[worst_top10_keys[i]]
#         img[side*(row+1):side*(row+2), side*col:side*(col+1)] = noisytestX[worst_top10_keys[i]]
#         img[side*(row+2):side*(row+3), side*col:side*(col+1)] = decoded[worst_top10_keys[i]]
#         img[side*(row+3):side*(row+4), side*col:side*(col+1)] = N[worst_top10_keys[i]]
#
#     img *= 255
#     img = img.astype(np.uint8)
#
#     #Save the image decoded
#     print("\nSaving results for worst after being encoded and decoded: @")
#     print(basepath+'/worst/')
#     io.imsave(basepath+'/worst/'+str(lamda)+'_cae_decode.png', img)
#
#
#     return
def evalPred(predX, trueX, trueY):

    trueX = np.reshape(trueX, (len(trueX),dim))
    predX = np.reshape(predX, (len(predX),dim))



    if predX.shape[1] > 1:
        recErr = ((predX - trueX) ** 2).sum(axis = 1)
    else:
        recErr = predX

    # print "+++++++++++++++++++++++++++++++++++++++++++"
    # print trueY
    # print recErr
    # print "+++++++++++++++++++++++++++++++++++++++++++"
    ap  = average_precision_score(trueY, recErr)
    try:
        auc = roc_auc_score(trueY, recErr)
    except ValueError:
        pass
        auc=0


    prec = precAtK(recErr, trueY, K = 10)

    return (ap, auc,prec)
def precAtK(pred, trueY, K = None):

    if K is None:
        K = trueY.shape[0]

    # label top K largest predicted scores as +'ve
    idx = np.argsort(-pred)
    predLabel = -np.ones(trueY.shape)
    predLabel[idx[:K]] = 1

    #print(predLabel)

    prec = precision_score(trueY, predLabel)

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


# [X,Y]=prepare_cifar_data_with_anamolies(image_and_anamolies_train)
# [testX,testY]=prepare_cifar_data_with_anamolies(image_and_anamolies_train)
normal_dataPath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/new_data/StitchedData/normal/"
anomalies_dataPath= "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/new_data/StitchedData/anomalies/"
[X,Y] = prepare_scene_data_with_anomalies(normal_dataPath, anomalies_dataPath)
print X.shape, Y.shape

# Reshape X to suite the 4 channel
X = X.reshape([-1, side, side, 1])





#define lamda set
# lamda_set = [ 0.0,0.01,0.1,0.5,1.0, 10.0, 100.0]
lamda_set = [ 0.0]
mue = 0.0
TRIALS= 7
ap = np.zeros((TRIALS,))
auc = np.zeros((TRIALS,))
prec = np.zeros((TRIALS,))

# outer loop for lamda
for l in range(0,len(lamda_set)):
    # Learn the N using softthresholding technique
    N =  0
    lamda = lamda_set[l]

    N = compute_softhreshold(X,N,lamda,X)

    XTrue = X
    #Predict the conv_AE autoencoder output
    XTrue = np.reshape(XTrue, (len(XTrue),side,side,channel))
    decoded = model.predict(XTrue)

    #compute MeanSqared error metric
    compute_mse(XTrue,decoded,lamda)

    # rank the best and worst reconstructed images
    [best_top10_keys,worst_top10_keys]=compute_best_worst_rank(XTrue,decoded)

    #Visualise the best and worst ( image, BG-image, FG-Image)
    XPred = np.reshape(np.asarray(decoded),(len(decoded),side,side,channel))
    visualise_anamolies_detected(XTrue,XTrue,decoded,N,best_top10_keys,worst_top10_keys,lamda)

    XPred = decoded

    YTrue = Y

    # YTrue = testY
    YTrue = label_binarize(YTrue, classes=[1, -1])

    print type(XPred), len(XPred)
    print type(XTrue), len(XTrue), XTrue.shape
    print type(YTrue), len(YTrue), YTrue.shape

    (ap[l], auc[l],prec[l]) = evalPred(XPred, XTrue, YTrue)
    print "AUPRC",l,ap[l]
    print "AUROC",l,auc[l]
    print "P@10",l,prec[l]
    # (ap[l], auc[l], prec[t], recErr) = evalPred(XPred, XTrue, YTrue)



print('AUPRC = %1.4f +- %1.4f' % (np.mean(ap), np.std(ap)/np.sqrt(TRIALS)))
print('AUROC = %1.4f +- %1.4f' % (np.mean(auc), np.std(auc)/np.sqrt(TRIALS)))
print('P@10  = %1.4f +- %1.4f' % (np.mean(prec), np.std(prec)/np.sqrt(TRIALS)))


# plotting the mean precision score
print("\n Saving the Mean square error Score ((Xclean, Xdecoded):")
fig1_mean_square_error=plt.figure(figsize=(8,5))
plt.xlabel("CAE-Denoiser")
plt.ylabel("Mean- Sq Error")
print("\n Mean square error Score ((Xclean, Xdecoded):")
print(mean_square_error_dict.values())
for k,v in mean_square_error_dict.iteritems():
    print k,v
# basic plot
data = mean_square_error_dict.values()
plt.boxplot(data)
fig1_mean_square_error.savefig(basepath+'_mean_square_error.png')


