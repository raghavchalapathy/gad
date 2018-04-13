from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tflearn.datasets import cifar10
from cifar10_utils import cifar10_data,prepare_cifar_data_with_anamolies
import numpy as np
from sklearn.metrics import average_precision_score,mean_squared_error
from skimage import io
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

basepath = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/experiments/cats_n_dogs/aae/examples/aae/"
class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        self.decoder = self.build_decoder()
        self.decoder.compile(loss=['mse'],
            optimizer=optimizer)

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder
        encoder = Sequential()

        encoder.add(Flatten(input_shape=self.img_shape))
        encoder.add(Dense(512))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(512))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.encoded_dim))

        encoder.summary()

        img = Input(shape=self.img_shape)
        encoded_repr = encoder(img)

        return Model(img, encoded_repr)

    def build_decoder(self):
        # Decoder
        decoder = Sequential()

        decoder.add(Dense(512, input_dim=self.encoded_dim))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(512))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(np.prod(self.img_shape), activation='tanh'))
        decoder.add(Reshape(self.img_shape))

        decoder.summary()

        encoded_repr = Input(shape=(self.encoded_dim,))
        gen_img = decoder(encoded_repr)

        return Model(encoded_repr, gen_img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.encoded_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.encoded_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        batch_size=5050

        ROOT = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/data/cifar-10_data/cifar-10-batches-py"

        (X, Y), (testX, testY) = cifar10.load_data(ROOT)
        image_and_inverted_anamolies_train = {'image1': 5, 'image2': 3, 'anomalies1': 3, 'imagecount': 2500,
                                              'anomaliesCount': 0, 'flipimage': "no", 'slice_stich': "no"}
        [X_train, Y] = prepare_cifar_data_with_anamolies(X, Y, image_and_inverted_anamolies_train)
        self.y_train = Y
        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Generate a half batch of embedded images
            latent_fake = self.encoder.predict(imgs)

            latent_real = np.random.normal(size=(half_batch, self.encoded_dim))

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            # Select a whole set of training samples
            # idx = len(X_train)-1
            imgs = X_train[idx]
            print("++++++++++++++++++++++++++++++++++++++++++++++")
            print(len(imgs),idx,X_train.shape,imgs.shape)
            print("++++++++++++++++++++++++++++++++++++++++++++++")

            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # # If at save interval => save generated image samples
            # if epoch % save_interval == 0:
            #     # Select a random half batch of images
            #     idx = np.random.randint(0, X_train.shape[0], 25)
            #     imgs = X_train[idx]
            #     self.save_imgs(epoch, imgs)

        encoded_imgs = self.encoder.predict(imgs)
        self.train_imgs = imgs
        self.decoded_imgs = self.decoder.predict(encoded_imgs)

    def save_imgs(self, epoch, imgs):
        r, c = 5, 5

        encoded_imgs = self.encoder.predict(imgs)
        gen_imgs = self.decoder.predict(encoded_imgs)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("aae/images/cifar10_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "aae/saved_model/%s.json" % model_name
            weights_path = "aae/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")





if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=1, batch_size=32, save_interval=200)
    print('='*35)
    print (aae.decoded_imgs.shape)
    decoded = aae.decoded_imgs
    x_test= aae.train_imgs
    y_test= aae.y_train
    print('=' * 35)
    print
    "Decoded Shape:", decoded.shape
    print
    "x_test Shape:", x_test.shape
    mean_square_error_dict = {}


    def compute_mse(Xclean, Xdecoded):
        # print len(Xdecoded)
        Xclean = np.reshape(Xclean, (len(Xclean), 3072))
        m, n = Xclean.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

        meanSq_error = mean_squared_error(Xclean, Xdecoded)
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


    def visualise_anamolies_detected(testX, decoded, best_top10_keys, worst_top10_keys):
        side = 32
        channel = 3
        # Display the decoded Original, noisy, reconstructed images
        print
        "side:", side
        print
        "channel:", channel
        img = np.ndarray(shape=(side * 2, side * 10, channel))
        print
        "img shape:", img.shape

        for i in range(10):
            row = i // 10 * 3
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
            # img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[best_top10_keys[i]]
            # img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[best_top10_keys[i]]

        img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for best after being encoded and decoded: @")
        print(basepath + '/results/best/')
        io.imsave(basepath + '/results/best/' + '_aae_best.png', img)

        # Display the decoded Original, noisy, reconstructed images for worst
        print("+++++++++++++++++++++++++++++++++++++++++++")
        print("Worst reconstruction Keys:", worst_top10_keys)
        print("+++++++++++++++++++++++++++++++++++++++++++")
        img = np.ndarray(shape=(side * 2, side * 10, channel))
        for i in range(10):
            row = i // 10 * 3
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
            # img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[worst_top10_keys[i]]
            # img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[worst_top10_keys[i]]

        # img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(basepath + '/results/worst/')
        io.imsave(basepath + '/results/worst/' + '_aae_worst.png', img)

        return


    def evalPred(predX, trueX, trueY):

        trueX = np.reshape(trueX, (len(trueX), 3072))
        predX = np.reshape(predX, (len(predX), 3072))

        if predX.shape[1] > 1:
            print
            "predX.shape[1]> 1"
            print
            len(predX)
            # recErr = ((predX - trueX) ** 2).sum(axis = 1)
            recErr = np.ones(len(predX))
            recErr[5000:5050] = -1
        else:
            recErr = predX

        print
        "+++++++++++++++++++++++++++++++++++++++++++"
        print
        len(trueY), trueY
        print
        len(recErr), recErr
        print
        "+++++++++++++++++++++++++++++++++++++++++++"

        ap = average_precision_score(trueY, recErr)
        auc = roc_auc_score(trueY, recErr)

        print
        "auprc:", ap
        print
        "auc:", auc

        # prec = precAtK(recErr, trueY, K = 10)
        prec = precision_score(trueY, recErr, average=None)

        # prec = mapk(trueX,predX)

        return (ap, auc, prec)


    # compute MeanSqared error metric
    compute_mse(x_test, decoded)
    # rank the best and worst reconstructed images
    [best_top10_keys, worst_top10_keys] = compute_best_worst_rank(x_test, decoded)

    # # Visualise the best and worst ( image, BG-image, FG-Image)
    visualise_anamolies_detected(x_test, decoded, best_top10_keys, worst_top10_keys)
    (aprc_val, auc_val, prec_val) = evalPred(decoded, x_test, y_test)

    print("++++++++++++++++++++++++++++++++++++++")
    print("AUPRC", aprc_val)
    print("AUROC", auc_val)
    print("P@10", prec_val)
    print("++++++++++++++++++++++++++++++++++++++")
