
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
    def __init__(self, datapath, prefix ):
        self.datapath = datapath
        self.prefix = prefix
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
        self.embedded_start_states = []
        self.controls = []
        self.end_states = []
        self.embedded_end_states = []

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
                    # self.start_states.pop()
                    i = 0
                    # if idx > 5000:
                    #     print "done Reading"
                    #     break
                    continue
                # Split Values in line and append to individual lists
                vals = line.split(',')
                if i % 5 == 0:
                    self.embedded_start_states.append([float(val) for val in vals])
                elif i % 5 == 1:
                    self.start_states.append([float(val) for val in vals])
                elif i % 5 == 2:
                    self.controls.append([float(val) for val in vals])
                elif i % 5 == 3:
                    self.embedded_end_states.append([float(val) for val in vals])
                elif i % 5 == 4:
                    # if np.sum(np.abs(np.asarray(self.embedded_end_states, dtype=np.float32)[-1, 1:4] -
                    #           np.asarray(self.embedded_start_states, dtype=np.float32)[-1, 1:4])) < 5:
                    #     self.controls.pop()
                    #     self.start_states.pop()
                    #     self.embedded_start_states.pop()
                    #     self.embedded_end_states.pop()
                    # elif np.abs(np.asarray(self.start_states, dtype=np.float32)[-1, 9:26]).any()>1:
                    #     print "rejecting: ", [val for val in vals]
                    #     print "length: ", len(vals)
                    #     self.controls.pop()
                    #     self.start_states.pop()
                    #     self.embedded_start_states.pop()
                    #     self.embedded_end_states.pop()
                    # else:
                    self.end_states.append([float(val) for val in vals])
                    idx += 1
                i += 1

        self.embedded_start_states = np.asarray(self.embedded_start_states, dtype=np.float32)
        self.controls = np.asarray(self.controls, dtype=np.float32)
        print self.embedded_start_states.shape
        print self.controls.shape
        self.start_states = np.asarray(self.start_states, dtype=np.float32)[:, 0:91]
        self.embedded_end_states = np.asarray(self.embedded_end_states, dtype=np.float32)
        self.end_states = np.asarray(self.end_states, dtype=np.float32)[:, 0:91]

        # Concatenate (1) normalized start states and (2) planar movement class
        # in order to have our desired input data
        # X = self.start_states
        # X = self.normalize_data(X)

        vel_deltas = self.end_states[:, 7:13] - self.start_states[:, 7:13]
        # mvmt_class = np.zeros((vel_deltas.shape[0],1))
        # for i, x in enumerate(vel_deltas):
        #     if x[0] > 0 and x[1] > 0 and np.abs(x[0]) < np.abs(x[1]):
        #         mvmt_class[i] = 1
        #     elif x[0] > 0 and x[1] > 0 and np.abs(x[0]) > np.abs(x[1]):
        #         mvmt_class[i] = 2
        #     elif x[0] > 0 and x[1] < 0 and np.abs(x[0]) < np.abs(x[1]):
        #         mvmt_class[i] = 3
        #     elif x[0] > 0 and x[1] < 0 and np.abs(x[0]) > np.abs(x[1]):
        #         mvmt_class[i] = 4
        #     elif x[0] < 0 and x[1] < 0 and np.abs(x[0]) < np.abs(x[1]):
        #         mvmt_class[i] = 5
        #     elif x[0] < 0 and x[1] < 0 and np.abs(x[0]) > np.abs(x[1]):
        #         mvmt_class[i] = 6
        #     elif x[0] < 0 and x[1] > 0 and np.abs(x[0]) < np.abs(x[1]):
        #         mvmt_class[i] = 7
        #     elif x[0] < 0 and x[1] > 0 and np.abs(x[0]) > np.abs(x[1]):
        #         mvmt_class[i] = 8
        #     else:
        #         print "ERROR FUX"
        
        self.vel_deltas = np.asarray(vel_deltas, dtype=np.float32)
        # self.mvmt_class = np.asarray(mvmt_class, dtype=np.float32)

        # Processing: Classify start configuration into 16 classes for snake
        #   Based on relative configuration quadrants of the links
        # pos_deltas = np.concatenate((self.embedded_start_states[:, 4:6],
        #                              self.embedded_start_states[:, 7:9]),
        #                             axis=1)
        # conf_class = np.zeros((pos_deltas.shape[0],1))
        # for i, x in enumerate(pos_deltas):
        #     if x[0] > 0 and x[1] > 0:
        #         if x[2] > 0 and x[3] > 0:
        #             conf_class[i] = 0
        #         elif x[2] > 0 and x[3] < 0:
        #             conf_class[i] = 1
        #         elif x[2] < 0 and x[3] < 0:
        #             conf_class[i] = 2
        #         elif x[2] < 0 and x[3] > 0:
        #             conf_class[i] = 3
        #     elif x[0] > 0 and x[1] < 0:
        #         if x[2] > 0 and x[3] > 0:
        #             conf_class[i] = 4
        #         elif x[2] > 0 and x[3] < 0:
        #             conf_class[i] = 5
        #         elif x[2] < 0 and x[3] < 0:
        #             conf_class[i] = 6
        #         elif x[2] < 0 and x[3] > 0:
        #             conf_class[i] = 7            
        #     elif x[0] < 0 and x[1] < 0:
        #         if x[2] > 0 and x[3] > 0:
        #             conf_class[i] = 8
        #         elif x[2] > 0 and x[3] < 0:
        #             conf_class[i] = 9
        #         elif x[2] < 0 and x[3] < 0:
        #             conf_class[i] = 10
        #         elif x[2] < 0 and x[3] > 0:
        #             conf_class[i] = 11
        #     elif x[0] < 0 and x[1] > 0:
        #         if x[2] > 0 and x[3] > 0:
        #             conf_class[i] = 12
        #         elif x[2] > 0 and x[3] < 0:
        #             conf_class[i] = 13
        #         elif x[2] < 0 and x[3] < 0:
        #             conf_class[i] = 14
        #         elif x[2] < 0 and x[3] > 0:
        #             conf_class[i] = 15

        # X = np.concatenate((np.asarray(conf_class, dtype=np.float32),
        #                     np.asarray(mvmt_class, dtype=np.float32)), axis=1)
        # y = self.normalize_labels()

        # indices = np.linspace(X.shape[0]-1,0,X.shape[0]-1)
        # # indices = np.random.permutation(X.shape[0])
        # training_idx = indices[:X.shape[0]*0.9].astype(int)
        # testing_idx = indices[X.shape[0]*0.9:].astype(int)

        # self.train_data = X[training_idx, :]
        # self.train_labels = y[training_idx, :]

        # self.test_data = X[testing_idx, :]
        # self.test_labels = y[testing_idx, :]

        # outdir = "."
        # if self.prefix == "tensegrity":
        #     outdir += "/models/superball/"
        # elif self.prefix == "toycar":
        #     outdir += "/models/socar/"
        # elif self.prefix == "snake":
        #     outdir += "/models/snake/"

        # self.save_data_as_hdf5(outdir + self.prefix + '_train.hdf5', 
        #                        self.train_data, self.train_labels)
        # self.save_data_as_hdf5(outdir + self.prefix + '_test.hdf5', 
        #                        self.train_data, self.train_labels)

    def normalize_labels(self):
        # return self.controls
        # print "original array: ", np.asarray(self.controls[:,0]).shape
        # print "addition: ", np.asarray(self.controls[:,0] + self.controls[:, 1]).shape
        # print "max: ", np.asarray(np.maximum(np.asarray(self.controls[:, 0] + self.controls[:, 1]), np.asarray(-self.controls[:, 0] + self.controls[:, 1]))).shape
        # print "maxes: ", maxes_1.shape
        # print "mins: ", mins_1.shape

        maxes_1 = np.asarray(np.maximum(np.asarray(self.controls[:, 0] + self.controls[:, 1]), np.asarray(-self.controls[:, 0] + self.controls[:, 1]))).reshape(self.controls.shape[0],1)
        mins_1 = np.asarray(np.minimum(np.asarray(self.controls[:, 0] + self.controls[:, 1]), np.asarray(-self.controls[:, 0] + self.controls[:, 1]))).reshape(self.controls.shape[0],1)
        bounds_1 = np.concatenate((maxes_1,mins_1), axis=1)

        maxes_2 = np.asarray(np.maximum(np.asarray(self.controls[:, 4] + self.controls[:, 5]), np.asarray(-self.controls[:, 4] + self.controls[:, 5]))).reshape(self.controls.shape[0],1)
        mins_2 = np.asarray(np.minimum(np.asarray(self.controls[:, 4] + self.controls[:, 5]), np.asarray(-self.controls[:, 4] + self.controls[:, 5]))).reshape(self.controls.shape[0],1)
        bounds_2 = np.concatenate((maxes_2, mins_2), axis=1)

        maxes_3 = np.asarray(np.maximum(np.asarray(self.controls[:, 8] + self.controls[:, 9]), np.asarray(-self.controls[:, 8] + self.controls[:, 9]))).reshape(self.controls.shape[0],1)
        mins_3 = np.asarray(np.minimum(np.asarray(self.controls[:, 8] + self.controls[:, 9]), np.asarray(-self.controls[:, 8] + self.controls[:, 9]))).reshape(self.controls.shape[0],1)
        bounds_3 = np.concatenate((maxes_3, mins_3), axis=1)

        maxes_4 = np.asarray(np.maximum(np.asarray(self.controls[:, 12] + self.controls[:, 13]), np.asarray(-self.controls[:, 12] + self.controls[:, 13]))).reshape(self.controls.shape[0],1)
        mins_4 = np.asarray(np.minimum(np.asarray(self.controls[:, 12] + self.controls[:, 13]), np.asarray(-self.controls[:, 12] + self.controls[:, 13]))).reshape(self.controls.shape[0],1)
        bounds_4 = np.concatenate((maxes_4, mins_4), axis=1)

        return np.concatenate((maxes_1, mins_1, maxes_2, mins_2,
                               maxes_3, mins_3, maxes_4, mins_4), axis=1)

    def unnormalize_data(self, data):
        desired_min = -1
        desired_max = 1
        desired_rng = desired_max - desired_min

        data = data - desired_min
        for i in range(0, data.shape[1]-1):
            data[:, i] = data[:, i] * (self.max[i] - self.min[i]) \
                        / desired_rng + self.min[i] + self.mean[i]

        return data

    def normalize_data(self, data):
        '''
        Normalize data to zero mean on [-1,1] interval for all dimensions
        '''
        self.mean = np.mean(data, axis=0)

        # do not substract duration
        # self.mean[-1] = 0

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
        for i in range(0, data.shape[1]):
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
            data[:, i] = (data[:, i] - self.min[i]) * (desiredMax - desiredMin)\
                / (self.max[i] - self.min[i]) + desiredMin

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
        binaryproto_file = open(self.prefix + string + '.binaryproto', 'wb')
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
        if self.prefix == "tensegrity":
            PROJECT_HOME += "/models/superball"
        elif self.prefix == "toycar":
            PROJECT_HOME += "/models/socar"
        elif self.prefix == "snake":
            PROJECT_HOME += "/models/snake"

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
        binaryproto_file = open(PROJECT_HOME + '/' + self.prefix + 
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
        self.embedded_end_states = model.embedded_end_states
        self.end_states = model.end_states
        self.start_states = model.start_states
        self.embedded_start_states = model.embedded_start_states
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
        # caffe.set_mode_gpu()
        solver = caffe.get_solver(self.solverpath)
        solver.solve()

    def test(self, datalines=[]):
        '''
        Performs Testing of the specified network
        '''
        if datalines == []:
            datalines = self.test_data
        outputs = np.zeros((1, self.test_labels.shape[1]))
        for data in datalines:
            # print "outputs: ", outputs.shape
            data_in = np.asarray([data])
            # print "data in: ", data_in.shape
            output_line = self.get_predicted_outputs(data_in)[0]
            # print "data out: ", output_line.shape
            outputs = np.append(outputs,
                                output_line,
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
        self.embedded_end_states = model.embedded_end_states
        self.end_states = model.end_states
        self.start_states = model.start_states
        self.embedded_start_states = model.embedded_start_states
        self.train_data = model.train_data
        self.train_labels = model.train_labels
        self.test_data = model.test_data
        self.test_labels = model.test_labels
        # self.mvmt_class = model.mvmt_class

    def train(self, data=[], labels=[]):
        np.random.seed(101)

        if data == []:
            data = self.train_data[:10000, :]
            print "No data explicitly passed to train(); Using 10k samples."
        if labels == []:
            labels = self.train_labels[:10000, :]

        X = np.array(data)
        Y = np.array(labels)
        
        rbf = GPy.kern.RBF(X.shape[1])
        self.model = GPy.models.SparseGPRegression(X, Y, kernel=rbf, 
                                                   num_inducing=250)
        
        self.model.optimize('tnc', messages=1, max_iters=100)

    def test(self, data):
        return self.model.predict(data)
