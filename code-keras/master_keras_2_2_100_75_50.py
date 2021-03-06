i0 = 2
i1 = 2
lay_neuron = [100, 75, 50]

#####################################
from cls_keras import ClsFuns
import random
import sklearn
import numpy
import matplotlib.pyplot
import matplotlib
import math
import h5py
import random
import pickle
import os

#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou.mat')


#####################################

matplotlib.__version__
sklearn.__version__

filename  = os.path.join(dir_data_python,'data.npy')
data_all  = numpy.load(filename ,allow_pickle=True)
data_all  = data_all.item()

filename  = os.path.join(dir_data_python,'ind.npy')
ind_all   = numpy.load(filename,allow_pickle=True)
ind_all   = ind_all.item()

datain    = data_all['input_data']
dataou    = data_all['output_data']
ind_train = ind_all['ind_train']
ind_test  = ind_all['ind_test']

cls_fun   = ClsFuns(i0,i1,lay_neuron)
cls_fun.fun_run(datain,dataou,ind_train,ind_test)
cls_fun.fun_losscurve()