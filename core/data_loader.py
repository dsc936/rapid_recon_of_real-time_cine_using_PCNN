import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import os
import hdf5storage
import pywt
from scipy import misc, ndimage
import random

class Loader2D(Dataset):
    def __init__(self, x_files, y_files, imsize = 256, mode = "complex", augment_flag = False):
        super(Loader2D, self).__init__()
        self.mode = mode
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        #self.augment_flag = augment_flag
        #self.aug_seq = iaa.Sequential([iaa.Fliplr(0.5),iaa.Rot90([0,1,2,3])])

    def crop(self, data, imsize):
        nx = data.shape[0]
        ny = data.shape[1]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return np.squeeze(data[resx: nx-resx, resy: ny-resy])

    def normalize(self, data, scale):
        nx = data.shape[0]//4
        data_center = self.crop(data,nx)
        return data/np.percentile(np.abs(data_center),scale)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        x = self.normalize(x,95)
        y = self.normalize(y,95)
        # x = x/np.percentile(np.abs(x), 98)
        # y = y/np.percentile(np.abs(y), 98)
        if self.mode == "complex":
            x = np.stack([x.real, x.imag], axis = 2)
            y = np.stack([y.real, y.imag], axis = 2)

        if self.mode == "complex":
            x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
            y = np.ascontiguousarray(np.transpose(y, (2, 0, 1)))
        elif self.mode =="mag":
            x = np.abs(x)[np.newaxis]
            y = np.abs(y)[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.FloatTensor(y)}

class Loader2D_segmentation(Dataset):
    def __init__(self, x_files, y_files, imsize = 144):
        super(Loader2D_segmentation, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize

    def crop(self, data, imsize):
        nx = data.shape[0]
        ny = data.shape[1]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return np.squeeze(data[resx: nx-resx, resy: ny-resy])

    def normalize(self, data):
        data = data - np.amin(data)
        nx = data.shape[0]//2
        data_center = self.crop(data,nx)
        # return data/np.amax(data_center)
        return data/np.percentile(np.abs(data_center),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))
        x = self.crop(np.abs(x), self.imsize)
        y = self.crop(np.abs(y), self.imsize)
        x = self.normalize(x)
        x = x[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

class Loader3D_segment(Dataset):
    def __init__(self, x_files, y_files, imsize = 144, t_slices = -1):
        super(Loader3D_segment, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices

    def crop(self, data, imsize):
        nx = data.shape[1]
        ny = data.shape[2]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return data[...,resx: nx-resx, resy: ny-resy]

    def normalize(self, data):
        nx = data.shape[1]//2
        data_center = self.crop(data,nx)
        # return data/np.amax(data_center)
        return data/np.percentile(np.abs(data_center),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        if (x.shape[0]//8)*8 < x.shape[0]:
            self.t_slices = (x.shape[0]//8)*8
        if self.t_slices != -1:
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]

        x = self.normalize(x)[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

class Loader3D_TPM(Dataset):
    def __init__(self, x_files, y_files, imsize = (128,96), t_slices = -1, mode = None):
        super(Loader3D_TPM, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices
        self.mode = mode

    def crop(self, data, imsize):
        nx = data.shape[-2]
        ny = data.shape[-1]
        padx = 0
        pady = 0
        if len(data.shape) == 3:
            output = np.zeros((data.shape[0],imsize,imsize))
        else:
            output = np.zeros((data.shape[0],data.shape[1],imsize,imsize))

        if nx>imsize:
            resx = (nx-imsize)//2
            data = data[...,resx: nx-resx,:]
        else:
            padx = (imsize-nx)//2
        if ny>imsize:
            resy = (ny-imsize)//2
            data = data[...,:,resy: ny-resy]
        else:
            pady = (imsize-ny)//2
        output[...,padx:imsize-padx,pady:imsize-pady] = data
        return output

    def normalize(self, data):
        nx = data.shape[2]//2
        data_center = self.crop(data,nx)
        return data/np.percentile(data_center,95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        x = np.transpose(x, (3, 2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = self.crop(x, self.imsize)
        x[0] = self.normalize(x[0])
        y = self.crop(y, self.imsize)

        if self.t_slices != -1:
            x = x[:,0:self.t_slices]
            y = y[0:self.t_slices]
        if self.mode =="mag":
            x_mag = x[0]
            x = x_mag[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.LongTensor(y)}

class Loader3D_TPM_test(Dataset):
    def __init__(self, x_files, imsize = 144, t_slices = -1, mode = None):
        super(Loader3D_TPM_test, self).__init__()
        self.x_files = x_files
        self.imsize = imsize
        self.t_slices = t_slices
        self.mode = mode

    def crop(self, data, imsize):
        nx = data.shape[-2]
        ny = data.shape[-1]
        padx = 0
        pady = 0
        if len(data.shape) == 3:
            output = np.zeros((data.shape[0],imsize,imsize))
        else:
            output = np.zeros((data.shape[0],data.shape[1],imsize,imsize))

        if nx>imsize:
            resx = (nx-imsize)//2
            data = data[...,resx: nx-resx,:]
        else:
            padx = (imsize-nx)//2
        if ny>imsize:
            resy = (ny-imsize)//2
            data = data[...,:,resy: ny-resy]
        else:
            pady = (imsize-ny)//2
        output[...,padx:imsize-padx,pady:imsize-pady] = data
        return output

    def normalize(self, data):
        nx = data.shape[2]//2
        data_center = self.crop(data,nx)
        return data/np.percentile(np.abs(data_center),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        x = np.transpose(x, (3, 2, 0, 1))
        x = self.crop(x, self.imsize)

        if self.t_slices != -1:
            x = x[:,0:self.t_slices]
        if self.mode =="mag":
            x_mag = x[0]
            x = x_mag[np.newaxis]
        return {"x": torch.FloatTensor(x)}
        
class Loader2D_wavelet(Dataset):
    def __init__(self, x_files, y_files, imsize = 256):
        super(Loader2D_wavelet, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize

    def crop(self, data, imsize):
        nx = data.shape[1]
        res = (nx-imsize)//2
        return np.squeeze(data[:,res: nx-res, res: nx-res])

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.array(list(hdf5storage.loadmat(self.x_files[idx]).values()))
        y = np.array(list(hdf5storage.loadmat(self.y_files[idx]).values()))

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        x = np.abs(x)/np.amax(np.abs(x), keepdims= True)
        y = np.abs(y)/np.amax(np.abs(y), keepdims= True)
        # x = (x + 1.0)/2.0
        # y = (y + 1.0)/2.0
        coeffs_x = pywt.wavedec2(x,'db1',level=1)
        DCx,(LH1x,HL1x,HH1x) = coeffs_x

        coeffs_y = pywt.wavedec2(y,'db1',level=1)
        DCy,(LH1y,HL1y,HH1y) = coeffs_y

        x = np.stack([DCx,LH1x,HL1x,HH1x], axis = 0)
        y = np.stack([DCy,LH1y,HL1y,HH1y], axis = 0)

        return {"x": torch.FloatTensor(x), "y": torch.FloatTensor(y)}

class Loader3D(Dataset):
    def __init__(self, x_files, y_files, imsize = 144, t_slices = -1):
        super(Loader3D, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices

    def crop(self, data, imsize):
        nx = data.shape[1]
        ny = data.shape[2]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return data[:,resx: nx-resx, resy: ny-resy]

    def normalize(self, data):
    	# result = np.abs(data)/np.amax(np.abs(data), keepdims= True)
    	# return (result*2.0-1.0)
        # nx = data.shape[1]//2
        # data_center = self.crop(data,nx)
        return np.abs(data)/np.percentile(np.abs(data),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        nz = x.shape[2]

        x = np.transpose(np.abs(x), (2, 0, 1))
        y = np.transpose(np.abs(y), (2, 0, 1))

        x = x/np.percentile(np.abs(y),95)
        y = y/np.percentile(np.abs(y),95)

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        if self.t_slices == -1:
            self.t_slices = (x.shape[0]//4)*4
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]
        else: 
            # frame = random.randint(0,nz-self.t_slices)
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]

        x = x[np.newaxis]
        y = y[np.newaxis]
        return {"x": torch.FloatTensor(x), "y": torch.FloatTensor(y)}

class Loader3D_complex(Dataset):
    def __init__(self, x_files, y_files, imsize = 144, t_slices = -1):
        super(Loader3D_complex, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices

    def crop(self, data, imsize):
        nx = data.shape[1]
        ny = data.shape[2]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return data[:,resx: nx-resx, resy: ny-resy]

    def normalize(self, data):
        # nx = data.shape[1]//2
        # data_center = self.crop(data,nx)
        return data/np.percentile(np.abs(data),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        nz = x.shape[2]

        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = x/np.percentile(np.abs(y),95)
        y = y/np.percentile(np.abs(y),95)

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        if self.t_slices == -1:
            self.t_slices = (x.shape[0]//4)*4
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]
        else: 
            # frame = random.randint(0,nz-self.t_slices)
            x = x[0: self.t_slices]
            y = y[0: self.t_slices]

        x = x[np.newaxis]
        y = y[np.newaxis]

        x = np.stack([x.real, x.imag], axis = 0)
        y = np.stack([y.real, y.imag], axis = 0)
        return {"x": torch.FloatTensor(x), "y": torch.FloatTensor(y)}

class Loader3D_complex_new(Dataset):
    def __init__(self, x_files, y_files, imsize = 144, t_slices = -1):
        super(Loader3D_complex_new, self).__init__()
        self.x_files = x_files
        self.y_files = y_files
        self.imsize = imsize
        self.t_slices = t_slices

    def crop(self, data, imsize):
        nx = data.shape[1]
        ny = data.shape[2]
        resx = 0
        resy = 0
        if nx>imsize:
            resx = (nx-imsize)//2
        if ny>imsize:
            resy = (ny-imsize)//2
        return data[:,resx: nx-resx, resy: ny-resy]

    def normalize(self, data):
        # nx = data.shape[1]//2
        # data_center = self.crop(data,nx)
        return data/np.percentile(np.abs(data),95)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):

        x = np.squeeze(np.array(list(hdf5storage.loadmat(self.x_files[idx]).values())))
        y = np.squeeze(np.array(list(hdf5storage.loadmat(self.y_files[idx]).values())))

        nz = x.shape[2]

        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        x = x/np.percentile(np.abs(y),95)
        y = y/np.percentile(np.abs(y),95)

        x = self.crop(x, self.imsize)
        y = self.crop(y, self.imsize)

        if self.t_slices == -1:
            self.t_slices = (x.shape[0]//4)*4
            x = x[0:self.t_slices]
            y = y[0:self.t_slices]
        else: 
            # frame = random.randint(0,nz-self.t_slices)
            x = x[0: self.t_slices]
            y = y[0: self.t_slices]

        x = np.stack([x.real, x.imag], axis = 3)
        y = np.stack([y.real, y.imag], axis = 3)

        x = x[np.newaxis]
        y = y[np.newaxis]

        return {"x": torch.FloatTensor(x), "y": torch.FloatTensor(y)}