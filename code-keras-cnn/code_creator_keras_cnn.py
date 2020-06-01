import os

#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras-cnn')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou.mat')

#########################

#$ random sampling and training-testing split rates
rtt = range(0,2)
rrs = range(0,2)

#$ define network architectures (only number of neuronsin each layer)
lay_neuron_all =  [[512,256,192,128,96,64,48,32,24],\
		   [512,],\
				   [256,192,128,96,64,48,32,24],\
				   [256,192,128,96,64,48,32,24],\
				   [64,48,32,24],\
				   [96,64],\
				   [256],\
				   [128],\
				   [64],\
				   [32],\
				   [16],\
				   [192,128,96,64,48,32,24]]

# create master codes
for lay_neuron in lay_neuron_all:

	tail_val = ''
	for jj in range(len(lay_neuron)):
		tail_val = tail_val + str(lay_neuron[jj])+ '_'

	tail_val = tail_val[:-1]

	if os.path.exists("ja_submit_{}".format(tail_val)):
		os.remove("ja_submit_{}".format(tail_val))

	for i0 in rtt:
		for i1 in rrs:

			if os.path.exists("master_keras_cnn_{}_{}_{}.py".format(i0+1,i1+1,tail_val)):
				os.remove("master_keras_cnn_{}_{}_{}.py".format(i0+1,i1+1,tail_val))

			if os.path.exists("ja_{}_{}_{}".format(i0+1,i1+1,tail_val)):
				os.remove("ja_{}_{}_{}".format(i0+1,i1+1,tail_val))

			f  = open("master_keras_cnn_{}_{}_{}.py".format(i0+1,i1+1,tail_val), "x")
			f0 = open("code_info_keras_cnn.txt","r")

			f.write("i0 = {}".format(i0 + 1))
			f.write("\ni1 = {}".format(i1 + 1))
			f.write("\nlay_neuron = {}".format(lay_neuron))

			f.write("\n{}".format(f0.read()))
			

