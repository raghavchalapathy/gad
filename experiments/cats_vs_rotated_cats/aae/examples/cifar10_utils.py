import numpy as np
from keras.datasets import cifar10
from imgaug import augmenters as iaa

def cifar10_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)


def func_slice_stich(X_images,Y_labels,slice_and_stitch):

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

def prepare_cifar_data_with_anamolies_stitch(image_and_anamolies,slice_and_stitch):

    ROOT = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cifar-10-batches-py"
    # load cifar-10 data
    (X, Y), (testX, testY) = cifar10.load_data()
    original_labels = Y
    original = X


    image1_label = image_and_anamolies['image1']
    image2_label = image_and_anamolies['image2']

    imagecnt = image_and_anamolies['imagecount']
    anoimagecnt = image_and_anamolies['anomaliesCount']
    flipImage = image_and_anamolies['flipimage']
    slice_stich = image_and_anamolies['slice_stich']

    import numpy as np
    idx1 = np.where(original_labels ==image1_label)
    idx2 = np.where(original_labels == image2_label)
    idx3 = np.where(original_labels == image2_label)
    idx4 = np.where(original_labels == image2_label)

    idx1 = idx1[0][:imagecnt]
    idx2 = idx2[0][:imagecnt]
    idx3 = idx3[0][:anoimagecnt]
    idx4 =  idx4[0][:anoimagecnt]




    images1 = original[idx1]
    images2 = original[idx2]
    ano_imgs = original[idx3]
    test_imgs = original[idx4]

    images_labels1 = original_labels[idx1]
    images_labels2 = original_labels[idx2]
    anomalies_labels = original_labels[idx3]
    testimg_labels = original_labels[idx4]

    temp_images = np.concatenate((images1, images2), axis=0)
    temp_images_label = np.concatenate((images_labels1, images_labels2), axis=0)
    data_train = temp_images  # Training set contains only cat




    if(flipImage == "yes"):
        seq = iaa.Sequential([
               iaa.Flipud(0.9)  # vertically flip 90% of the images
        ])
        ana_images = ano_imgs
        ana_images *= 255
        ana_images = ana_images.astype(np.uint8)
        ana_images = seq.augment_images(ana_images)
        # ana_labels = np.full(len(ana_images), -3)
        ana_labels = np.negative(anomalies_labels)
        data_test = np.concatenate((test_imgs, ana_images), axis=0) # data test contains normal and inverted cats
        # labels for these images
        print temp_images_label.shape
        print ana_labels.shape
        datalabels = np.concatenate((testimg_labels, ana_labels), axis=0)
        # from  scipy.io import savemat
        # mdict = {}
        # mdict.update({'imgs':data,'labels':datalabels})
        # savemat("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cats_n_invertedCats.mat", mdict)

        # return [data, datalabels]
        return cifar10_process(data_train), cifar10_process(data_test)

    if(slice_stich == "yes"):

           [ana_images,ana_labels] = func_slice_stich(original, original_labels,slice_and_stitch)
           data_train = temp_images
           data_test = np.concatenate((test_imgs, ana_images), axis=0)
           # labels for these images
           print len(temp_images_label),len(ana_labels)
           # datalabels = np.concatenate((temp_images_label, ana_labels), axis=0)
           # return [data, datalabels]
           return cifar10_process(data_train), cifar10_process(data_test)

    else:
        # Read the anomalies from file to numpy array and assign the labels for anomalies as 99
        from PIL import Image
        import os, numpy as np
        folder = '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/grp1anomalies'
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
        return [data, datalabels]


    return


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
        folder = '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/grp1anomalies'
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
        return [data, datalabels]


    return

