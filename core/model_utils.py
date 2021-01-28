import numpy as np
import torch
from collections import OrderedDict
import os

def save_model(model, optimizer = None, path = "/", filename = 'check_point.pth'):
	if optimizer is not None:
		torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, os.path.join(path, filename))
	else:
		torch.save({'model':model.state_dict(), 'optimizer':None}, os.path.join(path, filename))
	print("model saved as: " + os.path.join(path, filename))

def data_parallel_state_dict(state_dict, mode):
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = ""
		if mode == "single" and k.startswith("module."):
			name = k[7:]

		elif mode == "parallel" and not k.startswith("module."):
			name = "module."+k

		else:
			name = k

		new_state_dict[name] = v

	return new_state_dict

def load_model(model, path, model_name, mode = "parallel", optimizer = None, device = None):
	state_dict = torch.load(os.path.join(path, model_name))["model"]
	new_state_dict = data_parallel_state_dict(state_dict, mode)

	model.load_state_dict(new_state_dict)
	if device is not "None":
		model.to(device)

	print("loaded: " + os.path.join(path, model_name))
