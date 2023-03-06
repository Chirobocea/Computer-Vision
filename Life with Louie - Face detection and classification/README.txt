1. the libraries required to run the project including the full version of each library

python==3.9.13

opencv-python==4.6.0.66
numpy==1.23.4
scikit-image==0.19.3
scikit-learn==1.2.0
torch==1.13.1
torchvision==0.14.1
Pillow==9.3.0


For training ResNet18 was used:
RTX 3070 8GB

python==3.10.9
torch==1.13.1+cu116


2. how to run and where to look for the output file.

paths to change:
	script: Parameters.py [Code directory]
	change:	self.base_dir			to	root directory of this file on your machine	
			self.dir_test_examples		to	the absolute path for your test images
	


script: Run.py	->	running this code will solve both task 1 and task 2
output: [Solutions directory]

Notes:
The [Utilies directory] containes code used for prepairing traing data. 
This code is not necessary to test the model on your test data.
		