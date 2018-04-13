# path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_01_test/20171220-083741/imgs/"

#non-inductive
# path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_01_test/20171220-200636/imgs/"

#cats_ndogs_inductive ( Tested where the dogs were from different test set)
#path="/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171224-150031/imgs"

# Same Training set result sent to sanjay is below
# path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171224-191145/imgs/"
# Set below uses dogs and cats in both folder a and folder b with dogs taken from different distribution.
# path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171225-132437/imgs/"
# same mappings
# path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171225-140938/imgs"
#path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171225-152610/imgs"

# With Inverted cats
path = "/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/20171227-193655/imgs"

from skimage.measure import compare_ssim as ssim
import glob
import os

inpA_files = glob.glob( path+"/inputA*.jpg")
inpB_files = glob.glob( path+"/inputB*.jpg")
fakeA_files = glob.glob( path+"/fakeA*.jpg")
fakeB_files = glob.glob( path+"/fakeB*.jpg")
cycA_files = glob.glob( path+"/cycA*.jpg")
cycB_files = glob.glob( path+"/cycB*.jpg")
inpAlist = []
inpBlist = []
fakeAlist = []
fakeBlist = []
cycAlist = []
cycBlist = []
# print type(inpA_files)
for inpA,inpB,fakeA,fakeB,cycA,cycB in zip(inpA_files,inpB_files,fakeA_files,fakeB_files,cycA_files,cycB_files):
    headA,tailinpA = os.path.split(inpA)
    headB, tailinpB = os.path.split(inpB)
    headfakeA, tailfakeA = os.path.split(fakeA)
    headfakeB, tailfakeB = os.path.split(fakeB)
    headcycA, tailcycA = os.path.split(cycA)
    headcycB, tailcycB = os.path.split(cycB)
    # print tailinpA,tailinpB,tailfakeA,tailfakeB,tailcycA,tailcycB
    a,b = str(tailinpA).split("_0_")
    c, d = str(tailinpB).split("_0_")
    e, f = str(tailfakeA).split("_0_")
    g, h = str(tailfakeB).split("_0_")
    i, j = str(tailcycA).split("_0_")
    k, l = str(tailcycB).split("_0_")
    inpAlist.append(b)
    inpBlist.append(d)
    fakeAlist.append(f)
    fakeBlist.append(h)
    cycAlist.append(j)
    cycBlist.append(l)
    # print a,c,e,g,i,k

# print len(inpAlist),len(inpBlist),len(fakeAlist),len(fakeBlist),len(cycAlist),len(cycBlist)
inpAlist= sorted(inpAlist)
inpBlist= sorted(inpBlist)
fakeAlist= sorted(fakeAlist)
fakeBlist= sorted(fakeBlist)
cycAlist=sorted(cycAlist)
cycBlist= sorted(cycBlist)
import numpy as np
import PIL
error_A_FakeA = {}
error_A_CycA = {}
error_B_FakeB = {}
error_B_CycB = {}
from  sklearn.metrics import mean_squared_error
for filename in inpAlist:
    A = np.asarray(PIL.Image.open(path+"/inputA_0_"+filename))
    FakeA = np.asarray(PIL.Image.open(path + "/fakeA_0_" + filename))
    CycleA = np.asarray(PIL.Image.open(path + "/cycA_0_" + filename))
    # A = A[:, :, 0]
    # FakeA = FakeA[:, :, 0]
    # CycleA = CycleA[:, :, 0]
    # print A.shape,len(A)
    # print FakeA.shape
    # print CycleA.shape
    A = np.reshape(A,(196608))
    FakeA = np.reshape(FakeA, (196608))
    CycleA = np.reshape(CycleA, (196608))


    # error_A_FakeA.update({filename:mean_squared_error(A,FakeA)})
    # error_A_CycA.update({filename:mean_squared_error(A,CycleA)})

    error_A_FakeA.update({filename: np.linalg.norm(A-FakeA,ord=2)})
    error_A_CycA.update({filename: np.linalg.norm(A-CycleA,ord=2)})

    # error_A_FakeA.update({filename: ssim(A, FakeA)})
    # error_A_CycA.update({filename: ssim(A, CycleA)})


    B = np.asarray(PIL.Image.open(path+"/inputB_0_"+filename))
    FakeB = np.asarray(PIL.Image.open(path + "/fakeB_0_" + filename))
    CycleB = np.asarray(PIL.Image.open(path + "/cycB_0_" + filename))
    # B = B[:, :, 0]
    # FakeB = FakeB[:, :, 0]
    # CycleB = CycleB[:, :, 0]
    B = np.reshape(B,(196608))
    FakeB = np.reshape(FakeB, (196608))
    CycleB = np.reshape(CycleB, ( 196608))
    # error_B_FakeB.update({filename:mean_squared_error(B,FakeB)})
    # error_B_CycB.update({filename:mean_squared_error(B,CycleB)})

    error_B_FakeB.update({filename:np.linalg.norm(B-FakeB,ord=1)})
    error_B_CycB.update({filename:np.linalg.norm(B-CycleB,ord=1)})

    # error_B_FakeB.update({filename: ssim(B, FakeB)})
    # error_B_CycB.update({filename: ssim(B, CycleB)})

worst_sorted_A_FakeA = sorted(error_A_FakeA, key=error_A_FakeA.get, reverse=True)
worst_sorted_A_CycA = sorted(error_A_CycA, key=error_A_CycA.get, reverse=True)

worst_sorted_B_FakeB = sorted(error_B_FakeB, key=error_B_FakeB.get, reverse=True)
worst_sorted_B_CycB = sorted(error_B_CycB, key=error_B_CycB.get, reverse=True)

# print worst_sorted_A_FakeA[0:10]
# print worst_sorted_A_CycA[0:10]
# print worst_sorted_B_FakeB[0:10]
# print worst_sorted_B_CycB[0:10]

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar = ndar.astype(np.float64)
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Source : http://deeplearning.net/tutorial/utilities.html#how-to-plot
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype='uint8' if output_pixel_vals else out_array.dtype
                                              ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape,
                                                        tile_spacing, scale_rows_to_unit_interval,
                                                        output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                        # add the slice to the corresponding position in the
                        # output array
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


from  skimage import io
import  matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid



def getImageArrayFromKeys(worst):

    imglist = []
    print worst
    for filename in worst:
        imgs = np.asarray(PIL.Image.open(path + "/inputA_0_" + filename))
        # imgs = imgs[:, :, 0]

        imglist.append(imgs)
    imgArray = np.asarray(imglist)
    # imgs = np.reshape(imgArray, (len(imgArray), original_dim))

    return imgs


worst =  worst_sorted_B_CycB[0:30]
original_dim = 65536
img_Array = getImageArrayFromKeys(worst)
print img_Array.shape
#
# def plot_nxn(m,n, images):
#     images = images.reshape((m*n,256,256))
#     fig = plt.figure(1, (m, n))
#     grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(m, n),  # creates grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )
#
#     for i in range(n*n):
#         grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.
#
#     fig.savefig("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/cats_dogs_non_inductive_worstreconstructedImages_anomolies_detected.png")
#     plt.show()
#
# plot_nxn(5,10,img_Array)
#
#
# def plot_images(images, cls_true, cls_pred=None):
#     if len(images) == 0:
#         print("no images to show")
#         return
#     else:
#         random_indices = random.sample(range(len(images)), min(len(images), 9))
#
#     images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])
#
#     # Create figure with 3x3 sub-plots.
#     fig, axes = plt.subplots(3, 3)
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     img_size = 256
#     num_channels = 3
#
#
#     for i, ax in enumerate(axes.flat):
#         # Plot image.
#         ax.imshow(images[i].reshape(img_size, img_size, num_channels))
#
#         # Show true and predicted classes.
#         if cls_pred is None:
#             xlabel = "True: {0}".format(cls_true[i])
#         else:
#             xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
#
#         # Show the classes as the label on the x-axis.
#         ax.set_xlabel(xlabel)
#
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()
img_size = 256
num_channels = 3

import numpy as np
import matplotlib.pyplot as plt

# worst = ['15.jpg', '15.jpg', '15.jpg', '15.jpg', '10.jpg', '11.jpg', '15.jpg', '12.jpg', '15.jpg', '15.jpg', '15.jpg',
#          '17.jpg', '35.jpg', '25.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg', '5.jpg',
#          '5.jpg',
#          '5.jpg', '16.jpg', '16.jpg', '16.jpg', '16.jpg', '16.jpg']


def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    print height, width, intensity
    nrows = nindex // ncols
    print nindex
    print nrows * ncols
    print "====="
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result
def make_array(worst):
    imglist = []
    print worst
    for filename in worst:
        from PIL import Image
        imgs = np.asarray(Image.open(path + "/inputA_0_" + filename).convert('RGB'))
        imglist.append(imgs)
    imgArray = np.asarray(imglist)

    return imgArray


array = make_array(worst)
result = gallery(array)
plt.imshow(result)
plt.axis('off')
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
plt.savefig("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/L1_cats_with_dogs_epoch.png")
# plt.show()


#
# def plot_images(images, cls_true, cls_pred=None):
#     if len(images) == 0:
#         print("no images to show")
#         return
#     else:
#         random_indices = (range(len(images)), min(len(images), 9))
#
#     images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])
#
#     # Create figure with 3x3 sub-plots.
#     fig, axes = plt.subplots(3, 3)
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#
#     for i, ax in enumerate(axes.flat):
#         # Plot image.
#         ax.imshow(images[i].reshape(img_size, img_size, num_channels))
#
#         # Show true and predicted classes.
#         if cls_pred is None:
#             xlabel = "True: {0}".format(cls_true[i])
#         else:
#             xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
#
#         # Show the classes as the label on the x-axis.
#         ax.set_xlabel(xlabel)
#
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()
#
# # plot_images(img_Array, cls_true, cls_pred=None)

#
# result_normal = tile_raster_images(img_Array, [256, 256], [5, 10])
#
# print "Saving results.."
# io.imsave("/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/output/cyclegan/exp_03_test/cats_dogs_non_inductive_worstreconstructedImages"+'.png', result_normal)
