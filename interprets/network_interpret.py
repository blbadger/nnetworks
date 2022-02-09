# network_interpret.py

import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 


class Interpret:

	def __init__(self, model, file):
		self.model = model
		self.file = file
		self.fields_ls = ['store_id', 
						'market_id', 
						'created_at',
						'total_busy_dashers', 
						'total_onshift_dashers', 
						'total_outstanding_orders',
						'estimated_store_to_consumer_driving_duration',
						'linear_ests']

	def occlusion(self):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input: torch.Tensor
			output: torch.Tensor
			field_array: arr[int], indicies that mark ends of each field 

		Returns:
			occlusion_arr: array[float] of scores per input index
			indicies_arr: array[int] 

		"""

		validation_data = Format(self.file, training=True)
		occl_size = 1

		output, input, output_tensor, input_tensor = validation_data.validation()

		output_tensor = self.model(input_tensor)

		zeros_tensor = torch.zeros(input_tensor[:occl_size].shape)
		indicies_arr = []
		total_index = 0
		taken_ls = [4, 1, 4, 3, 3, 3, 4, 4]
		occlusion_arr = [0 for i in taken_ls]
		start = 0
		end = 0

		for i in range(len(taken_ls)):
			end += taken_ls[i]

			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)

			input_copy[start:end + 1][:] = zeros_tensor
			output_missing = self.model(input_copy)
			occlusion_arr[i] = abs(float(output_missing) - float(output_tensor))
			indicies_arr.append(i)
			start += taken_ls[i]


		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return indicies_arr, occlusion_arr


	def graph_attributions(self):
		"""
		Plot the attributions of the model for 5 inputs

		Args:
			None

		Returns:
			None (dislays matplotlib.pyplot object)
		
		"""

		# view horizontal bar charts of occlusion attributions for five input examples
		for i in range(5):
			indicies, occlusion = self.occlusion()
			plt.style.use('dark_background')
			plt.barh(indicies, occlusion)
			plt.yticks(np.arange(0, len(self.fields_ls)), [i for i in self.fields_ls])
			plt.tight_layout()
			plt.show()
			plt.close()


class Interpret:

	def __init__(self, model, input_tensors, output_tensors, fields):
		self.model = model 
		self.field_array = fields
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors


	def occlusion(self, input_tensor, field_array):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
			field_array: arr[int], indicies that mark ends of each field 

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""

		occl_size = 1

		output = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor)
		occlusion_arr = [0 for i in range(len(input_tensor))]
		indicies_arr = []
		total_index = 0

		for i in range(len(field_array)):

			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			for j in range(total_index, total_index + field_array[i]):
				input_copy[j] = 0.

			total_index += field_array[i]

			output_missing = self.model(input_copy)

			# assumes a 1-dimensional output
			occlusion = abs(float(output) - float(output_missing))
			indicies_arr.append(i)

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return indicies_arr, occlusion_arr


	def gradientxinput(self, input_tensor, output_shape, model):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# change output to float
		input.requires_grad = True
		output = model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % 67 ==0 and i > 0: # assumes ASCII character set
				saliency_arr.append(s)
				s = 0
			s += float(final[i])

		# append final element
		saliency_arr.append(s)

		# max norm
		for i in range(len(inputxgrad)):
			maximum = max(inputxgrad[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(inputxgrad)):
				inputxgrad[i] /= maximum

		return inputxgrad


	def heatmap(self, n_observed=100, method='combined'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_array = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_array.append(attributions)

		plt.imshow(attributions_array)
		plt.savefig('attributions')
		plt.close()
		
		return














