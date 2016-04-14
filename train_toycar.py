'''Requirements:
 - Caffe (script to install Caffe and pycaffe on a new Ubuntu 14.04 LTS x64 or Ubuntu 14.10 x64. 
   CPU only, multi-threaded Caffe. http://stackoverflow.com/a/31396229/395857)
 - sudo pip install pydot
 - sudo apt-get install -y graphviz

Interesting resources on Caffe:
 - https://github.com/BVLC/caffe/tree/master/examples
 - http://nbviewer.ipython.org/github/joyofdata/joyofdata-articles\
 /blob/master/deeplearning-with-caffe/\
 Neural-Networks-with-Caffe-on-the-GPU.ipynb
'''

import subprocess
import platform
import copy
import sys
import os

# from sklearn.datasets import load_iris
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import h5py
import caffe
import caffe.draw
import google.protobuf 
import GPy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Model:
    '''
    Data containers and Visualization functions for models
    '''
    def __init__(self, datapath ):
        self.datapath = datapath
        self.load_data()

    def plot2d(self, x, y, c='r'):
        c = plt.scatter(x, y, c=c)
        plt.colorbar(c)
        plt.show()

    def plot3d(self, x, y, z, sz=2):
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.plot(x, y, z, 'o', ms=sz)
        plt.show()

    def load_data(self):
        '''
        Load Data into class for training and testing
        '''
        self.start_states = []
        self.controls = []
        self.durations = []
        self.end_states = []

        with open(self.datapath, 'r') as infile:
            data = infile.readlines()
            idx, i = 0, 0

            # Stop at end of file
            for line in data:
                if line == '':
                    print "end"
                    break
                # Reset and continue at Trajectory break
                if len(line) == 1:
                    self.start_states.pop()
                    i = 0
                    if idx > 1000000:
                        print "done Reading"
                        break
                    continue
                # Split Values in line and append to individual lists
                vals = line.split(',')
                if i % 3 == 0:
                    self.start_states.append([float(val) for val in vals])
                    if i != 0:
                        self.end_states.append([float(val) for val in vals])
                        idx += 1
                elif i % 3 == 1:
                    self.controls.append([float(val) for val in vals])
                elif i % 3 == 2:
                    self.durations.append([float(val) for val in vals])
                i += 1

        self.start_states = np.asarray(self.start_states, dtype=np.float32)
        self.end_states = np.asarray(self.end_states, dtype=np.float32)
        self.controls = np.asarray(self.controls, dtype=np.float32)
        self.durations = np.asarray(self.durations, dtype=np.float32)

        print "start states: ", self.start_states.shape
        for i in range(0, len(self.start_states)):
            if np.abs(self.end_states[i, 2] - self.start_states[i, 2]) > 5:
                if self.end_states[i, 2] > 0:
                    self.end_states[i, 2] -= 2*np.pi
                else:
                    self.end_states[i, 2] += 2*np.pi            

        X = np.concatenate((self.start_states, self.controls, self.durations),
                           axis=1)

        X = self.normalize_data(X)
        y = self.normalize_labels()

        indices = np.random.permutation(X.shape[0])
        training_idx = indices[:X.shape[0]*0.9]
        testing_idx = indices[X.shape[0]*0.9:]

        self.train_data = X[training_idx, :]
        self.train_labels = y[training_idx, :]

        self.test_data = X[900000:1000000, :]
        self.test_labels = y[900000:1000000, :]

        self.save_data_as_hdf5('toycar_train.hdf5', 
                               self.train_data, self.train_labels)
        self.save_data_as_hdf5('toycar_test.hdf5', 
                               self.train_data, self.train_labels)

    def normalize_labels(self):
        '''
        Normalize end states such that each state variable x_i
        is now represented by \delta x_i
        '''
        y = self.end_states[:, :] - self.start_states[:, :]

        return y

    def unnormalize_data(self, data):
        desired_min = -1
        desired_max = 1
        desired_rng = desired_max - desired_min

        data = data - desired_min
        for i in range(0, data.shape[1]):
            data[:, i] = data[:, i] * (self.max[i] - self.min[i]) \
                        / desired_rng + self.min[i] + self.mean[i]

        return data

    def normalize_data(self, data):
        '''
        Normalize data to zero mean on [-1,1] interval for all dimensions
        '''
        self.mean = np.mean(data, axis=0)

        # do not substract duration
        self.mean[7] = 0

        # Mean Shift
        data = data - self.mean

        # Find bounds, define desired bounds
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

        # Write 'em Out
        self.save_binaryproto(self.mean, ftype="mean")
        self.save_binaryproto(self.min, ftype="min")
        self.save_binaryproto(self.max, ftype="max")

        desiredMin = -1
        desiredMax = 1

        # Normalize
        for i in range(0, 7):
            data[:, i] = (data[:, i] - self.min[i]) * (desiredMax - desiredMin)\
                / (self.max[i] - self.min[i]) + desiredMin

        return data

    def normalize_test_data(self, data):
        '''
        Normalize data to zero mean on [-1,1] interval for all dimensions
        '''
        data = data - self.mean
        desiredMin = -1
        desiredMax = 1

        # Normalize
        for i in range(0, data.shape[1]):
            data[:, i] = (data[:, i] - self.min[:, i]) * (desiredMax - desiredMin)\
                / (self.max[:, i] - self.min[:, i]) + desiredMin

        return data

    def save_data_as_hdf5(self, filename_hdf5, data, labels):
        '''
        HDF5 is one of the data formats Caffe accepts
        '''
        with h5py.File(filename_hdf5, 'w') as f:
            f['data'] = data.astype(np.float32)
            f['label'] = labels.astype(np.float32)

    def write_binaryproto(self, data, string):
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.channels = data.shape[0]
        blob.data.extend(data.astype(float).flat)
        binaryproto_file = open('toycar_' + string + '.binaryproto', 'wb')
        binaryproto_file.write(blob.SerializeToString())
        binaryproto_file.close()

    def save_binaryproto(self, data, ftype='mean'):
        '''
        Take the mean values of the raw data and store them as binaryproto type
        In order to use them later for deploy normalization
        '''
        # Convert to 32bit float
        data = np.array(data, dtype=np.float32)

        # Set project home dir
        PROJECT_HOME = os.path.abspath('.')

        # Initialize blob to store serialized means
        blob = caffe.proto.caffe_pb2.BlobProto()

        # Custom dimensions for blob for this project
        blob.num = 1
        blob.channels = data.shape[0]
        blob.height = 1
        blob.width = 1

        # Reshape data and copy into blob\n",
        blob.data.extend(data.astype(float).flat)

        # Write file
        binaryproto_file = open(PROJECT_HOME + '/toycar_' +
                                ftype + '.binaryproto', 'wb')
        binaryproto_file.write(blob.SerializeToString())
        binaryproto_file.close()


class Network(Model):
    '''
    Initialize Network instance with arg structure:
    <datapath, solverpath, structurepath, deploypath, weightspath>
    '''
    def __init__(self, datapath, solverpath, 
                 structurepath, deploypath, 
                 weightspath=os.path.abspath('.') + 
                             '2fc_iter_100000.caffemodel',):
        '''
        Initialize ANN with existing model
        '''
        self.datapath = datapath
        self.solverpath = solverpath
        self.structurepath = structurepath
        self.deploypath = deploypath
        self.weightspath = weightspath
        self.load_data()

    def __init__(self, model, solverpath, 
                 structurepath, deploypath, 
                 weightspath=os.path.abspath('.') + 
                             '2fc_iter_100000.caffemodel',):
        '''
        Initialize ANN with existing model
        '''
        self.solverpath = solverpath
        self.structurepath = structurepath
        self.deploypath = deploypath
        self.weightspath = weightspath

        self.controls = model.controls
        self.durations = model.durations
        self.end_states = model.end_states
        self.start_states = model.start_states
        self.train_data = model.train_data
        self.train_labels = model.train_labels
        self.test_data = model.test_data
        self.test_labels = model.test_labels

    def print_network_parameters(self, net):
        '''
        Print the parameters of the network
        '''
        print(net)
        print('net.inputs: {0}'.format(net.inputs))
        print('net.outputs: {0}'.format(net.outputs))
        print('net.blobs: {0}'.format(net.blobs))
        print('net.params: {0}'.format(net.params))

    def print_network(self):
        '''
        Draw the ANN architecture
        '''
        _net = caffe.proto.caffe_pb2.NetParameter()
        f = open(self.structurepath)
        google.protobuf.text_format.Merge(f.read(), _net)
        caffe.draw.draw_net_to_file(_net, self.structurepath + '.png')
        print('Draw ANN done!')

    def print_network_weights(self):
        '''
        For each ANN layer, print weight heatmap and weight histogram
        '''
        net = caffe.Net(self.structurepath, self.weightspath, caffe.TEST)
        for layer_name in net.params:
            # weights heatmap
            arr = net.params[layer_name][0].data
            plt.clf()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            cax = ax.matshow(arr, interpolation='none')
            fig.colorbar(cax, orientation="horizontal")
            plt.savefig('{0}_weights_{1}.png'.format(self.weightspath,
                                                     layer_name),
                        dpi=100, format='png', bbox_inches='tight')
            plt.close()

            # weights histogram
            plt.clf()
            plt.hist(arr.tolist(), bins=20)
            # savefig: use format='svg' or 'pdf' for vectorial pictures
            plt.savefig('{0}_weights_hist_{1}.png'.format(self.weightspath,
                                                          layer_name), dpi=100,
                        format='png',
                        bbox_inches='tight')
            plt.close()

    def get_predicted_outputs(self, inputs):
        '''
        Get several predicted outputs
        '''
        outputs = []
        net = caffe.Net(self.deploypath, self.weightspath, caffe.TRAIN)
        out = net.forward(data=inputs)
        output = out[net.outputs[0]]
        outputs.append(copy.deepcopy(output))

        return outputs

    def train(self):
        '''
        Performs Training of the specified network and outputs PNG images
        showing resulting learned weights and histograms of weights
        '''
        # Train network
        caffe.set_mode_gpu()
        solver = caffe.get_solver(self.solverpath)
        solver.solve()

    def test(self, datalines=[]):
        '''
        Performs Testing of the specified network
        '''
        if datalines == []:
            datalines = self.test_data
        outputs = np.asarray([[0, 0, 0, 0, 0]])
        for data in datalines:
            data_in = np.asarray([data])
            outputs = np.append(outputs,
                                self.get_predicted_outputs(data_in)[0],
                                axis=0)
        outputs = outputs[1:, :]
        # print "Labels:"
        # print self.test_labels

        # print "Predictions: "
        # print outputs

        return outputs

    def euclidean_loss(pred, labels):
        '''
        Hand Calculate the Euclidean Loss to Compare with Model Output
        '''
        result = labels-pred
        size = pred.shape[0]

        loss = np.sum(np.square(result), axis=1)
        loss = np.sum(loss, axis=0) / (2 * size)

        print "Euclidean Loss: ", loss


class SGPRegression(Model):
    def __init__(self, model):
        self.controls = model.controls
        self.durations = model.durations
        self.end_states = model.end_states
        self.start_states = model.start_states
        self.train_data = model.train_data
        self.train_labels = model.train_labels
        self.test_data = model.test_data
        self.test_labels = model.test_labels

    def train(self, data=[], labels=[]):
        np.random.seed(101)

        if data == []:
            data = self.train_data[:10000, :]
        if labels == []:
            labels = self.train_labels[:10000, :]

        X = np.array(data)
        Y = np.array(labels)
        
        rbf = GPy.kern.RBF(8)
        self.model = GPy.models.SparseGPRegression(X, Y, kernel=rbf, 
                                                   num_inducing=250)
        
        self.model.optimize('tnc', messages=1, max_iters=100)

    def test(self, data):
        return self.model.predict(data)
