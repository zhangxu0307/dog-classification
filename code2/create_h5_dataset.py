# from tflearn.data_utils import build_hdf5_image_dataset
# import h5py
#
# new_train = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/new_train.txt"
# new_val = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/new_val.txt"
# new_test = "/home/adoke/tf_tutorial/aircrafts_new/from_start/a3_variants_test.txt"
#
# # image_shape option can be set to different values to create images of different sizes
# build_hdf5_image_dataset(new_val, image_shape=(224, 224), mode='file', output_path='new_val_224.h5', categorical_labels=True, normalize=False)
# print ('Done creating new_val.h5')
# build_hdf5_image_dataset(new_test, image_shape=(224, 224), mode='file', output_path='new_test_224.h5', categorical_labels=True, normalize=False)
# print ('Done creating new_test.h5')
# build_hdf5_image_dataset(new_train, image_shape=(488, 488), mode='file', output_path='new_train_488.h5', categorical_labels=True, normalize=False)
# print ('Done creating new_train_488.h5')


from tflearn.data_utils import build_hdf5_image_dataset

trainfile = "../data/train_data_crop/"
validfile = "../data/valid_data_crop/"
totaltrain = "../data/total_train_crop/"

# build_hdf5_image_dataset(trainfile, image_shape=(224, 224),
#                          mode='folder', output_path='../data/train.h5',
#                          categorical_labels=True, normalize=False, files_extension=['.jpg'])
# print('Done creating train.h5')
# build_hdf5_image_dataset(validfile, image_shape=(224, 224),
#                          mode='folder', output_path='../data/valid.h5',
#                          categorical_labels=True, normalize=False, files_extension=['.jpg'])
# print('Done creating valid.h5')

build_hdf5_image_dataset(totaltrain, image_shape=(224, 224),
                         mode='folder', output_path='../data/totaltrain_crop.h5',
                         categorical_labels=True, normalize=False, files_extension=['.jpg'])
print('Done creating totaltrain.h5')