import h5py
import cv2
with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/total/HKG_024_b_0023+Micheal+431.h5', 'r') as hdf5file:
    print  hdf5file['label'][()]
    my_array = hdf5file['image'][()]
    a=my_array[0,0:3,:,:]
    print my_array[0,1,:,:].shape
    cv2.imwrite('/home/lzz/1.jpg',my_array[0,0,:,:])
