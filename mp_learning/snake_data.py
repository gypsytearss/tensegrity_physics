
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

def normalize(angle, min_, max_):
    if angle < min_:
        return angle + 2*np.pi
    if angle > max_:
        return angle - 2*np.pi
    return angle

    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return normalize(np.arctan2(v1_u[0], v1_u[1]) - np.arctan2(v2_u[0], v2_u[1]), -3.1415, 3.1415)

class Model:
    '''
    Data containers and Visualization functions for models
    '''
    def __init__(self, datapath, prefix ):
        self.datapath = datapath
        self.prefix = prefix
        self.load_data()

    # 1 - Calculate slope of points, input: 1x91 snake state array
    def slope (self, idx1, idx2, m="no_debug" ):
        X, Y = [], []
        x_mid = self.states[idx1,idx2,39]
        y_mid = self.states[idx1,idx2,40]
        for i in [0,6]:
            X.append(self.states[idx1,idx2,i*13] )
            Y.append(self.states[idx1,idx2,(i*13)+1] )
        s_x = np.std(X)
        s_y = np.std(Y)
        corr_x_y = np.corrcoef(X,Y)
        slope = corr_x_y[0,1]*(s_y / s_x)
        quadrant = 0
        point = []
        if (X[0] > X[-1]) and (slope >= 0.0):
            quadrant = 1
            point = [1, slope]
        elif (X[0] < X[-1]) and (slope < 0.0):
            quadrant = 2
            point = [-1, -slope]
        elif (X[0] < X[-1]) and (slope >= 0.0):
            quadrant = 3
            point = [-1, -slope]
        elif (X[0] > X[-1]) and (slope < 0.0):
            quadrant = 4
            point = [1, slope]
        else:
            raise ValueError("Invalid slope/quadrant computation")
        if (m == "debug"):
            print "X: ",X
            print "Y: ",Y
            print "quad: ", quadrant
            print "slope: ", slope
            print "point: ", point
            point1 = [10, 10*slope]
            point2 = [-10, -10*slope]
            plot_snake_state_pts(X,Y, point1, point2)
        return point, quadrant

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
        self.states = []
        self.controls = []
        self.durations = []
        self.disp_angles = []
        self.disp = []
        self.disp_along_heading = []
        self.heading_change_norm = []
        self.heading_change = []

        with open(self.datapath, 'r') as infile:
            data = infile.readlines()
            idx, i = 0, 0
            state_seq = []

            # Stop at end of file
            for line in data:
                if line == '':
                    print "end"
                    break
                # Reset and continue at Trajectory break
                if len(line) == 1:
                    i = 0
                    seq = np.asarray(state_seq, dtype=np.float32)[:, 0:91]
                    self.states.append(seq)
                    state_seq = []
                    idx += 1
                    continue
                # Split Values in line and append to individual lists
                vals = line.split(',')
                if i == 0:
                    self.controls.append([float(val) for val in vals])
                elif i == 1:
                    self.durations.append([float(val) for val in vals])
                elif i >= 2:
                    state_seq.append([float(val) for val in vals])

                i += 1

        self.controls = np.asarray(self.controls, dtype=np.float32)
        self.durations = np.asarray(self.durations, dtype=np.float32)
        self.states = np.asarray(self.states, dtype=np.float32)

        # 2 - Remove displacement(sqrt(dx^2+dy^2) middle link) < 3.0
        disp_thresh = []
        for i,st in enumerate(self.states):
            disp_10 = np.sqrt(np.square(self.states[i,20,39] - self.states[i,10,39]) + 
                              np.square(self.states[i,20,40] - self.states[i,10,40]))
            if disp_10 > 5.0:
                disp_thresh.append(i)
        self.states = self.states[disp_thresh]
        self.controls = self.controls[disp_thresh]
        self.durations = self.durations[disp_thresh]

        # FILTER: Keep only stable gaits (>5 consecutive cycles within +/- 5 deg displacement)
        # COMPUTE: (1) disp: total diplacement (2) disp_angles: angle of disp of center link (3) disp_along_heading
        stable_idxs = []
        for i in range(0,self.states.shape[0]):
            cons_cycs = 0
            stable_val = 0.0
            for j in range(self.states.shape[1]-1,1,-1):
                start_disp = np.asarray(self.states[i,j-1,39:41])
                end_disp = np.asarray(self.states[i,j,39:41])
                
                start_slope, quadrant = self.slope(i,j-1)
                end_slope, quadrant = self.slope(i,j)
                
                start_vec = np.asarray(start_slope)
                end_vec = np.asarray((end_disp - start_disp)[0:2])
                
                disp_total = np.sqrt(np.sum(np.square(start_disp - end_disp)))
                disp_angle = angle_between(start_vec, end_vec)*180/np.pi
                
                slope_unit = unit_vector(np.asarray(end_slope))
                disp_along_heading = np.dot(end_vec,slope_unit)
                
                head_chg = angle_between(start_slope, end_slope)*180/np.pi
                
                if j == self.states.shape[1]-1:
                    stable_val = disp_angle
                    self.disp_along_heading.append(disp_along_heading * 1000.0 / self.durations[i])
                    self.disp_angles.append(disp_angle * 1000.0 / self.durations[i])
                    self.disp.append(disp_total * 1000.0 / self.durations[i])
                    self.heading_change.append(head_chg)
                    self.heading_change_norm.append(head_chg * 1000.0 / self.durations[i])
                elif ((disp_angle < stable_val + 5.0) and (disp_angle > stable_val - 5.0)):
                    cons_cycs += 1
                else:
                    break
            if cons_cycs >= 5:
                stable_idxs.append(i)
        self.disp_along_heading = np.asarray(self.disp_along_heading)
        self.disp_angles = np.asarray(self.disp_angles)
        self.disp = np.asarray(self.disp)
        self.heading_change_norm = np.asarray(self.heading_change_norm)
        self.heading_change = np.asarray(self.heading_change)

        self.disp = self.disp[stable_idxs]
        self.disp_angles = self.disp_angles[stable_idxs]
        self.disp_along_heading = self.disp_along_heading[stable_idxs]
        self.states = self.states[stable_idxs]
        self.controls = self.controls[stable_idxs]
        self.durations = self.durations[stable_idxs]
        self.heading_change = self.heading_change[stable_idxs]
        self.heading_change_norm = self.heading_change_norm[stable_idxs]

        # Create Training/Test data and split
        indices = np.linspace(self.controls.shape[0]-1,0,self.controls.shape[0]-1)
        training_idx = indices[:self.controls.shape[0]*0.9].astype(int)
        testing_idx = indices[self.controls.shape[0]*0.9:].astype(int)

        self.train_data = self.controls[training_idx, :]
        self.train_labels = self.heading_change_norm[training_idx, :]

        self.test_data = self.controls[testing_idx, :]
        self.test_labels = self.heading_change_norm[testing_idx, :]


    def normalize_labels(self):
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
    def __init__(self, data_model):
        self.data_model = data_model

    def train(self, data=[], labels=[]):
        np.random.seed(101)

        if data == []:
            data = self.data_model.train_data[:10000, :]
            print "No data explicitly passed to train(); Using 10k samples."
        if labels == []:
            labels = self.data_model.train_labels[:10000, :]

        X = np.array(data)
        Y = np.array(labels)
        
        rbf = GPy.kern.RBF(X.shape[1])
        self.model = GPy.models.SparseGPRegression(X, Y, kernel=rbf, 
                                                   num_inducing=500)
        
        self.model.optimize('tnc', messages=0, max_iters=500)

    def test(self, data):
        return self.model.predict(data)
