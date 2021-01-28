import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from .model_utils import load_model, save_model
import torchvision.models as models
from torch.autograd import Variable

def setDevice(device):
	if device is "None":
		device = torch.device("cpu")
	return device

class L1_Charbonnier_Loss(nn.Module):
	def __init__(self):
		super(L1_Charbonnier_Loss, self).__init__()
		self.eps = 1e-6
	def forward(self, x, y):
		diff = torch.add(x, -y)
		error = torch.sqrt(diff*diff+self.eps)
		loss = torch.mean(error)
		return loss	

class ComplexConv3D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3,stride = 1, padding = 1,bias = True):
		super(ComplexConv3D, self).__init__()
		self.conv_re = nn.Conv3d(in_channels, out_channels,kernel_size = kernel_size,stride = stride, padding = padding,bias =bias)
		self.conv_im = nn.Conv3d(in_channels, out_channels,kernel_size = kernel_size,stride = stride, padding = padding,bias =bias)
	def forward(self, x):
		real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
		imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
		output = torch.stack((real,imaginary),dim = 1)
		return output

class ComplexBN3D(nn.Module):
	def __init__(self, out_channels, affine = False, track_running_stats = False):
		super(ComplexBN3D, self).__init__()
		self.BN = nn.BatchNorm3d(out_channels, affine =affine,track_running_stats =track_running_stats)
	def forward(self, x):
		real = self.BN(x[:,0])
		imaginary = self.BN(x[:,1])
		output = torch.stack((real,imaginary),dim = 1)
		return output	

class ComplexReLU(nn.Module):
	def __init__(self, inplace = False):
		super(ComplexReLU, self).__init__()
		self.ReLU_re = nn.ReLU(inplace = inplace)
		self.ReLU_im = nn.ReLU(inplace = inplace)
	def forward(self, x):
		real = self.ReLU_re(x[:,0])
		imaginary = self.ReLU_im(x[:,1])
		output = torch.stack((real,imaginary),dim = 1)
		return output	

class ComplexConvTranspose3D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=2,stride = 2, padding = 0,bias = False):
		super(ComplexConvTranspose3D, self).__init__()
		self.convtrans = nn.ConvTranspose3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride = stride,bias =bias)
	def forward(self, x):
		real = self.convtrans(x[:,0]) 
		imaginary = self.convtrans(x[:,1])
		output = torch.stack((real,imaginary),dim = 1)
		return output

class ComplexMaxPool3D(nn.Module):
	def __init__(self,kernel_size = 2,stride = 2):
		super(ComplexMaxPool3D, self).__init__()
		self.MP = nn.MaxPool3d(kernel_size= kernel_size,stride = stride)
	def forward(self, x):
		real = self.MP(x[:,0])
		imaginary = self.MP(x[:,1])
		output = torch.stack((real,imaginary),dim = 1)
		return output	

class ComplexConvBlock3D(nn.Module):
	def __init__(self, in_channels, out_channels, drop_out = 0):
		super(ComplexConvBlock3D, self).__init__()
		self.drop_out = nn.Dropout3d(drop_out)
		self.relu = ComplexReLU()
		self.conv = ComplexConv3D(in_channels, out_channels,3, padding = 1, bias = False)
		self.bn = ComplexBN3D(out_channels)
	def forward(self, x):
		out = self.conv(x)
		out = self.drop_out(out)
		out = self.bn(out)
		out = self.relu(out)
		return out

class ComplexConvNet3D(nn.Module):
	def __init__(self,in_channels = 1, out_channels = 1, features = 16, blocks = 3, drop_out = 0, mode = None):
		super(ComplexConvNet3D, self).__init__()
		self.mode = mode
		self.conv1 = ComplexConv3D(in_channels, features, kernel_size = 3, stride = 1, padding = 1)
		self.bn = ComplexBN3D(features, affine = False, track_running_stats = False)
		self.relu = ComplexReLU()
		layers = []
		for i in range(1,blocks):
			layers.append(ComplexConvBlock3D(features, features, drop_out = drop_out))

		self.conv4 = ComplexConv3D(features, out_channels, kernel_size = 3, stride = 1, padding = 1)
		self.layer = nn.Sequential(*layers)
		
	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.layer(out)
		out = self.conv4(out)
		if self.mode =="res":
			return out + x
		else:
			return out

class ComplexUNet3D(nn.Module):
	def __init__(self,features = 16,drop_out = 0,mode = None):
		super(ComplexUNet3D, self).__init__()
		self.features = features
		self.mode = mode
		# self.weight = torch.nn.Parameter(torch.as_tensor(1.0))
		# self.weight.requires_grad = True
		self.max_pool = ComplexMaxPool3D((2,2,2),stride =(2,2,2))
		self.conv1 = ComplexConvBlock3D(features,features,drop_out)
		self.conv2 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv3 = ComplexConvBlock3D(features*4,features*4,drop_out)
		self.conv4 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv5 = ComplexConvBlock3D(features,features,drop_out)
		self.down1 = ComplexConvBlock3D(1,features)
		self.down2 = ComplexConvBlock3D(features,features*2,drop_out)
		self.down3 = ComplexConvBlock3D(features*2,features*4,drop_out)
		self.up_pool_1 = ComplexConvTranspose3D(features*4,features*2,(2,2,2),stride = (2,2,2))
		self.up_pool_2 = ComplexConvTranspose3D(features*2,features,(2,2,2),stride = (2,2,2))
		self.up1 = ComplexConvBlock3D(features*4,features*2,drop_out)
		self.up2 = ComplexConvBlock3D(features*2,features,drop_out)
		self.conv_res =  ComplexConvBlock3D(2,1,drop_out)
		self.conv_final = ComplexConv3D(features,1,3,padding = 1, stride=1, bias = False)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.down1(x)
		x1 = self.conv1(x1)
		x2 = self.max_pool(x1)
		x2 = self.down2(x2)
		x2 = self.conv2(x2)
		x3 = self.max_pool(x2)
		x3 = self.down3(x3)
		x3 = self.conv3(x3)
		x4 = self.up_pool_1(x3)
		x5 = torch.cat((x2,x4),2)
		x5 = self.up1(x5)
		x5 = self.conv4(x5)
		x6 = self.up_pool_2(x5)
		x7 = torch.cat((x1,x6),2)
		x7 = self.up2(x7)
		x7 = self.conv5(x7)
		out = self.conv_final(x7)
		if self.mode == "res":
			x8 = torch.cat((x,out),2)
			# return out + torch.mul(x,self.weight)
			return self.conv_res(x8)
		else:
			return out

class ComplexUNet3Dres(nn.Module):
	def __init__(self,features = 16,drop_out = 0,mode = None):
		super(ComplexUNet3Dres, self).__init__()
		self.features = features
		self.mode = mode
		self.max_pool = ComplexMaxPool3D((2,2,2),stride =(2,2,2))
		self.conv1 = ComplexConvBlock3D(features,features,drop_out)
		self.conv2 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv3 = ComplexConvBlock3D(features*4,features*4,drop_out)
		self.conv4 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv5 = ComplexConvBlock3D(features,features,drop_out)
		self.down1 = ComplexConvBlock3D(1,features)
		self.down2 = ComplexConvBlock3D(features,features*2,drop_out)
		self.down3 = ComplexConvBlock3D(features*2,features*4,drop_out)
		self.up_pool_1 = ComplexConvTranspose3D(features*4,features*2,(2,2,2),stride = (2,2,2))
		self.up_pool_2 = ComplexConvTranspose3D(features*2,features,(2,2,2),stride = (2,2,2))
		self.up1 = ComplexConvBlock3D(features*4,features*2,drop_out)
		self.up2 = ComplexConvBlock3D(features*2,features,drop_out)
		self.conv_final = ComplexConv3D(features,1,3,padding = 1, stride=1, bias = False)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.down1(x)
		x1 = self.conv1(x1)
		x2 = self.max_pool(x1)
		x2 = self.down2(x2)
		x2 = self.conv2(x2)
		x3 = self.max_pool(x2)
		x3 = self.down3(x3)
		x3 = self.conv3(x3)
		x4 = self.up_pool_1(x3)
		x5 = torch.cat((x2,x4),2)
		x5 = self.up1(x5)
		x5 = self.conv4(x5)
		x6 = self.up_pool_2(x5)
		x7 = torch.cat((x1,x6),2)
		x7 = self.up2(x7)
		x7 = self.conv5(x7)
		out = self.conv_final(x7)
		return out + torch.mul(x,0.2)


class DenseBlock(nn.Module):
	def __init__(self, in_channels, out_channels, drop_out = 0.1):
		super(DenseBlock, self).__init__()
		self.drop_out = nn.Dropout3d(drop_out)
		self.relu = ComplexReLU()
		self.conv = ComplexConv3D(in_channels, out_channels,3, padding = 1, bias = False)
		self.bn = ComplexBN3D(out_channels)
	def forward(self, x):
		out = self.conv(x)
		out = self.drop_out(out)
		out = self.bn(out)
		out = self.relu(out)
		return torch.cat([x,out],2)

class DenseUnet3D(nn.Module):
	def __init__(self,features = 16,drop_out = 0,mode = None):
		super(DenseUnet3D, self).__init__()
		self.mode = mode
		self.max_pool = ComplexMaxPool3D((1,2,2),stride =(1,2,2))
		self.conv_init = ComplexConvBlock3D(1,features)
		self.dense1 = DenseBlock(features,features,drop_out)
		self.dense2 = DenseBlock(features*2,features,drop_out)
		self.dense3 = DenseBlock(features*3,features,drop_out)
		self.up_pool_1 = ComplexConvTranspose3D(features*4,features*4,(1,2,2),stride = (1,2,2))
		self.up_pool_2 = ComplexConvTranspose3D(features*8,features*8,(1,2,2),stride = (1,2,2))
		self.dense4 = DenseBlock(features*7,features,drop_out)
		self.dense5 = DenseBlock(features*10,features,drop_out)
		self.conv_final = ComplexConv3D(features*11,1,3,padding = 1, stride=1, bias = False)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.conv_init(x)
		x1 = self.dense1(x1)
		x2 = self.max_pool(x1)
		x2 = self.dense2(x2)
		x3 = self.max_pool(x2)
		x3 = self.dense3(x3)
		x4 = self.up_pool_1(x3)
		x5 = torch.cat((x2,x4),2)
		x5 = self.dense4(x5)
		x6 = self.up_pool_2(x5)
		x7 = torch.cat((x1,x6),2)
		x7 = self.dense5(x7)
		out = self.conv_final(x7)
		return out

class ComplexUNet3D_2DPool(nn.Module):
	def __init__(self,features = 16,drop_out = 0,mode = None):
		super(ComplexUNet3D_2DPool, self).__init__()
		self.mode = mode
		self.max_pool = ComplexMaxPool3D((1,2,2),stride =(1,2,2))
		self.conv1 = ComplexConvBlock3D(features,features,drop_out)
		self.conv2 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv3 = ComplexConvBlock3D(features*4,features*4,drop_out)
		self.conv4 = ComplexConvBlock3D(features*2,features*2,drop_out)
		self.conv5 = ComplexConvBlock3D(features,features,drop_out)
		self.down1 = ComplexConvBlock3D(1,features)
		self.down2 = ComplexConvBlock3D(features,features*2,drop_out)
		self.down3 = ComplexConvBlock3D(features*2,features*4,drop_out)
		self.up_pool_1 = ComplexConvTranspose3D(features*4,features*2,(1,2,2),stride = (1,2,2))
		self.up_pool_2 = ComplexConvTranspose3D(features*2,features,(1,2,2),stride = (1,2,2))
		self.up1 = ComplexConvBlock3D(features*4,features*2,drop_out)
		self.up2 = ComplexConvBlock3D(features*2,features,drop_out)
		self.conv_final = ComplexConv3D(features,1,3,padding = 1, stride=1, bias = False)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.down1(x)
		x1 = self.conv1(x1)
		x2 = self.max_pool(x1)
		x2 = self.down2(x2)
		x2 = self.conv2(x2)
		x3 = self.max_pool(x2)
		x3 = self.down3(x3)
		x3 = self.conv3(x3)
		x4 = self.up_pool_1(x3)
		x5 = torch.cat((x2,x4),2)
		x5 = self.up1(x5)
		x5 = self.conv4(x5)
		x6 = self.up_pool_2(x5)
		x7 = torch.cat((x1,x6),2)
		x7 = self.up2(x7)
		x7 = self.conv5(x7)
		out = self.conv_final(x7)
		if self.mode == "res":
			return out + x
		else:
			return out

class ComplexDisc(nn.Module):
	def __init__(self, in_channels = 1, features = 32, drop_out = 0.0):
		super(ComplexDisc, self).__init__()
		self.features = features
		self.conv1 = ComplexConvBlock3D(in_channels,features)
		self.conv2 = ComplexConvBlock3D(features,features*2,drop_out)
		self.conv3 = ComplexConvBlock3D(features*2,features*4,drop_out)
		self.conv4 = ComplexConvBlock3D(features*4,features*2,drop_out)
		self.linear = nn.Linear(features*2,1) 

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)      
		x4 = self.conv4(x3) 
		x5 = x4.permute(0,1,3,4,5,2)
		x5 = x5.contiguous()
		x6 = x5.view(-1, self.features*2)
		out = self.linear(x6)
		return out

class FeatureExtractor(nn.Module):
	def __init__(self, num_layers = 15):
		super(FeatureExtractor, self).__init__()
		model = models.vgg16_bn(pretrained=True)
		self.feat_ext = self._feat_extractor(model,num_layers)

	def _feat_extractor(self, model, num_layers):
		for param in model.parameters():
			param.requires_grad = False
		feat_ext = torch.nn.Sequential(*(model.features[i] for i in range(num_layers)))
		return feat_ext

	def forward(self, x):
		inp = []
		for i in range(x.shape[3]):
			real = x[:,0,0:1,i,:,:]
			out_real = torch.cat([real,real,real],dim=1)
			inp.append(out_real)
			imag = x[:,1,0:1,i,:,:]
			out_imag = torch.cat([imag,imag,imag],dim=1)
			inp.append(out_imag)
		inp = torch.cat(inp, dim = 0)
		return self.feat_ext(inp)

class FeatureExtractor_time(nn.Module):
	def __init__(self, num_layers = 15):
		super(FeatureExtractor_time, self).__init__()
		model = models.vgg16_bn(pretrained=True)
		self.feat_ext = self._feat_extractor(model,num_layers)

	def _feat_extractor(self, model, num_layers):
		for param in model.parameters():
			param.requires_grad = False
		feat_ext = torch.nn.Sequential(*(model.features[i] for i in range(num_layers)))
		return feat_ext

	def forward(self, x):
		inp = []
		for i in range(x.shape[4]):
			real_xt = x[:,0,0:1,:,:,i]
			real_yt = x[:,0,0:1,:,i,:]
			out_real_xt = torch.cat([real_xt,real_xt,real_xt],dim=1)
			out_real_yt = torch.cat([real_yt,real_yt,real_yt],dim=1)
			inp.append(out_real_xt)
			inp.append(out_real_yt)

			imag_xt = x[:,1,0:1,:,:,i]
			imag_yt = x[:,1,0:1,:,i,:]
			out_imag_xt = torch.cat([imag_xt,imag_xt,imag_xt],dim=1)
			out_imag_yt = torch.cat([imag_yt,imag_yt,imag_yt],dim=1)
			inp.append(out_imag_xt)
			inp.append(out_imag_yt)			
		inp = torch.cat(inp, dim = 0)
		return self.feat_ext(inp)		

class ConvBlock3D(nn.Module):
	def __init__(self, in_channels, out_channels, drop_out = 0,mode = None):
		super(ConvBlock3D, self).__init__()
		self.mode = mode
		self.drop_out = nn.Dropout3d(drop_out)
		self.relu = nn.ReLU(inplace = False)
		self.conv = nn.Conv3d(in_channels, out_channels,3, padding = 1, bias = False)
		self.bn = nn.BatchNorm3d(out_channels, affine = False, track_running_stats = False)
	def forward(self, x):
		out = self.conv(x)
		out = self.drop_out(out)
		out = self.bn(out)
		out = self.relu(out)
		if self.mode =="res":
			return out + x
		else:
			return out

class UNet3D(nn.Module):
	def __init__(self,features = 32,drop_out = 0.0,mode = "none"):
		super(UNet3D, self).__init__()
		self.features = features
		self.mode = mode
		self.max_pool = nn.MaxPool3d((2,2,2),stride = (2,2,2))
		self.conv1 = ConvBlock3D(features,features,drop_out)
		self.conv2 = ConvBlock3D(features*2,features*2,drop_out)
		self.conv3 = ConvBlock3D(features*4,features*4,drop_out)
		self.conv4 = ConvBlock3D(features*2,features*2,drop_out)
		self.conv5 = ConvBlock3D(features,features,drop_out)
		self.down1 = ConvBlock3D(1,features)
		self.down2 = ConvBlock3D(features,features*2,drop_out)
		self.down3 = ConvBlock3D(features*2,features*4,drop_out)
		self.up_pool_1 = nn.ConvTranspose3d(features*4,features*2,(2,2,2),stride = (2,2,2))
		self.up_pool_2 = nn.ConvTranspose3d(features*2,features,(2,2,2),stride = (2,2,2))
		self.up1 = ConvBlock3D(features*4,features*2,drop_out)
		self.up2 = ConvBlock3D(features*2,features,drop_out)
		self.conv_final = nn.Conv3d(features,1,3,padding = 1, stride=1, bias = False)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		x1 = self.down1(x)
		x1 = self.conv1(x1)
		x2 = self.max_pool(x1)
		x2 = self.down2(x2)
		x2 = self.conv2(x2)
		x3 = self.max_pool(x2)
		x3 = self.down3(x3)
		x3 = self.conv3(x3)
		x4 = self.up_pool_1(x3)
		x5 = torch.cat((x2,x4),1)
		x5 = self.up1(x5)
		x5 = self.conv4(x5)
		x6 = self.up_pool_2(x5)
		x7 = torch.cat((x1,x6),1)
		x7 = self.up2(x7)
		x7 = self.conv5(x7)
		out = self.conv_final(x7)
		if self.mode == "res":
			return out + x
		else:
			return out

class ConvNet3D(nn.Module):
	def __init__(self,in_channels = 1, out_channels = 1, features = 64, blocks = 3):
		super(ConvNet3D, self).__init__()
		self.conv1 = nn.Conv3d(in_channels, features, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm3d(features, affine = False, track_running_stats = False)
		layers = []
		for i in range(1,blocks):
			layers.append(ConvBlock3D(features, features, drop_out = 0))

		self.conv4 = nn.Conv3d(features, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.layer = nn.Sequential(*layers)
		
	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = F.relu(out)
		out = self.layer(out)
		out = self.conv4(out)
		return out

class ReconNet3D(nn.Module):
	def __init__(self, mode = "dual", features = 64,blocks = 3):
		super(ReconNet3D, self).__init__()
		self.mode = mode
		self.base_network = ConvNet3D
		if self.mode == "dual":
			self.IS1 = self.base_network(features =features ,blocks = blocks)
			self.IS2 = self.base_network(features =features ,blocks = blocks)
		elif self.mode == "mag" or self.mode == "batch":
			self.IS1 = self.base_network(features =features ,blocks =blocks)
		elif self.mode == "channel":
			self.IS1 = self.base_network(2, 2,features =features ,blocks = blocks)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self, x):
		out = x
		if self.mode == "dual":
			out1 = self.IS1(out[:, 0:1])
			out2 = self.IS2(out[:, 1:])
			out = torch.cat([out1, out2], dim = 1)

		elif self.mode == "mag" or self.mode == "channel":
			out = self.IS1(out)

		elif self.mode == "batch":
			out1 = self.IS1(out[:, 0:1])
			out2 = self.IS1(out[:, 1:])
			out = torch.cat([out1, out2], dim = 1)
		return out
