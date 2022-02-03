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