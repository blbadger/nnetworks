# NN_prep_snippets.py

from PIL import Image
import image_slicer
import os


image_files = ['path/to/file.jpg', 'path/to/file2.png']

image_directory = '/home/bbadger/Desktop/Neural_networks'


class Imageprep:

	def __init__(self, image_file):
		self.image_files = image_files

	def slice(self, slices):
		"""
		Divides an image into smaller images of equal dimension (l//n by w//n)

		Args:
			slices: int, the number of desired smaller images

		Returns:
			None (appends sliced files to directory)

		"""

		for i, file in enumerate(self.image_files):
			im = Image.open(file)

			# save images as 'path/to/file_0_1.jpg' etc.
			image_slicer.slice('/path/to/file.jpg', slices)

		return
		

	def rename(self, image_directory):
		"""
		Rename files with a name that is easily indexed for future iteration.

		Args:
			image_directory: str, path to directory containing images

		Returns:
			None (modifies image names in-place)

		"""

		path = image_directory
		files = os.listdir(path)

		for index, file in enumerate(files):
			os.rename(os.path.join(path, file), os.path.join(path, ''.join(['s', str(index), '.png'])))

		return


	def rotate(self, image_directory):
		"""
		Rotate images from -5 to 5 degrees, 1 degree increments, saving each.

		Args:
			image_directory: str, path to directory of images

		Returns:
			None (saves images)

		"""

		path = image_directory
		files = os.listdir(path)

		for i, file in enumerate(files):
			im = Image.open(file)
			for j in range(-5,5):
				im.rotate(j)

				# reformat file name (assumes . only in file ending)
				pair_ls = [i for i in file.split('.')]
				core_name = pair_ls[0]
				ending = pair_ls[1]
				new_file_name = core_name + f'_{i}_{j}' + ending

				im.save(new_file_name)

		return



