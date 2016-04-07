'''
Requirements:
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
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import h5py
import caffe
import caffe.draw
import google.protobuf 


def load_data():
    '''
    Load Sample for Forward Pass from Toy Car Data set
    '''
    start_states, controls, durations, end_states = [], [], [], []

    with open('data_output_50Hz.txt', 'r') as infile:
        data = infile.readlines()

        idx, i = 0, 0

        for line in data:
            # Stop at end of file
            if line == '':
                break

            # Reset and continue at Trajectory break
            if len(line) == 1:
                start_states.pop()
                i = 0
                if idx > 1000000:
                    break
                continue
                
            # Split Values in line and append to individual lists
            vals = line.split(',')
            if i % 3 == 0:
                start_states.append([float(val) for val in vals])
                if i != 0:
                    end_states.append([float(val) for val in vals])
                    idx += 1
            elif i % 3 == 1:
                controls.append([float(val) for val in vals])
            elif i % 3 == 2:
                durations.append([float(val) for val in vals])
               
            i += 1

    X = np.concatenate((start_states, controls, durations), axis=1)
    start_states = np.asarray(start_states, dtype=np.float32)
    end_states = np.asarray(end_states, dtype=np.float32)
    
    X = normalize_data(X)
    y = end_states  # normalize_labels(start_states, end_states)

    # Shuffle the data around and split 3M training, ~750k validation
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:900000], indices[900000:1000000]
    
    train_X = X[training_idx, :]
    train_y = y[training_idx, :]

    test_X = X[test_idx, :]
    test_y = y[test_idx, :]

    return train_X, train_y, test_X, test_y


def write_binaryproto(data, string):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.channels = data.shape[0]
    blob.data.extend(data.astype(float).flat)
    binaryproto_file = open('toycar_' + string + '.binaryproto', 'wb')
    binaryproto_file.write(blob.SerializeToString())
    binaryproto_file.close()


def save_binaryproto(data, ftype='mean'):
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
    binaryproto_file = open(PROJECT_HOME + '/toycar_' + ftype + '.binaryproto', 'wb')
    binaryproto_file.write(blob.SerializeToString())
    binaryproto_file.close()



def normalize_labels(start_states, end_states):
    '''
    Normalize end states such that positional coordinates e.g. (x,y)
    are now represented by delta_(x,y)
    '''
    y = end_states
    y[:, 0] = end_states[:, 0] - start_states[:, 0]
    y[:, 1] = end_states[:, 1] - start_states[:, 1]

    return y


def unnormalize_data(data, meanx, minx, maxx):
    desired_min = -1
    desired_max = 1
    desired_rng = desired_max - desired_min
    
    data = data - desired_min
    for i in range(0, 7):
        data[:, i] = data[:, i] * (maxx[i] - minx[i]) \
                    / desired_rng + minx[i] + meanx[i]
        
    return data


def normalize_data(data):
    '''
    Normalize data to zero mean on [-1,1] interval for all dimensions
    '''
    X_mean = np.mean(data, axis=0)
    
    # do not substract duration
    X_mean[7] = 0
    
    # Mean Shift
    data_t = data - X_mean
    
    # Find bounds, define desired bounds
    X_min = np.min(data_t, axis=0)
    X_max = np.max(data_t, axis=0)

    # Write 'em Out
    save_binaryproto(X_mean, ftype="mean")
    save_binaryproto(X_min, ftype="min")
    save_binaryproto(X_max, ftype="max")

    desiredMin = -1
    desiredMax = 1
    
    # Normalize 
    for i in range(0, 7):
        data_t[:, i] = (data_t[:, i] - X_min[i]) * (desiredMax - desiredMin)\
            / (X_max[i] - X_min[i]) + desiredMin

    return data_t


def save_data_as_hdf5(hdf5_data_filename, data, labels):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = labels.astype(np.float32)


def train(solver_prototxt_filename):
    '''
    Train the ANN
    '''
    # caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()


def print_network_parameters(net):
    '''
    Print the parameters of the network
    '''
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params))    


def get_predicted_output(deploy_prototxt_filename, 
                         caffemodel_filename, input, net=None):
    '''
    Get the predicted output, i.e. perform a forward pass
    '''
    if net is None:
        net = caffe.Net(deploy_prototxt_filename, 
                        caffemodel_filename, caffe.TEST)
    
    print "Input: "
    print input 
    out = net.forward(data=input)
    return out[net.outputs[0]]


def print_network(prototxt_filename, caffemodel_filename):
    '''
    Draw the ANN architecture
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png')
    print('Draw ANN done!')


def print_network_weights(prototxt_filename, caffemodel_filename):
    '''
    For each ANN layer, print weight heatmap and weight histogram 
    '''
    net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST)
    for layer_name in net.params: 
        # weights heatmap 
        arr = net.params[layer_name][0].data
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(arr, interpolation='none')
        fig.colorbar(cax, orientation="horizontal")
        plt.savefig('{0}_weights_{1}.png'.format(caffemodel_filename, 
                                                 layer_name), 
                    dpi=100, format='png', bbox_inches='tight')
        plt.close()

        # weights histogram  
        plt.clf()
        plt.hist(arr.tolist(), bins=20)
        # savefig: use format='svg' or 'pdf' for vectorial pictures
        plt.savefig('{0}_weights_hist_{1}.png'.format(caffemodel_filename, 
                                                      layer_name), dpi=100, 
                    format='png', 
                    bbox_inches='tight')  
        plt.close()


def get_predicted_outputs(deploy_prototxt_filename, 
                          caffemodel_filename, inputs):
    '''
    Get several predicted outputs
    '''
    outputs = []
    net = caffe.Net(deploy_prototxt_filename, caffemodel_filename, caffe.TRAIN)
    outputs.append(copy.deepcopy(get_predicted_output(deploy_prototxt_filename, 
                                                      caffemodel_filename, 
                                                      inputs, net)))
    return outputs    


def get_accuracy(true_outputs, predicted_outputs):
    '''

    '''
    number_of_samples = true_outputs.shape[0]
    number_of_outputs = true_outputs.shape[1]
    threshold = 0.0  # 0 if SigmoidCrossEntropyLoss ; 0.5 if EuclideanLoss
    for output_number in range(number_of_outputs):
        predicted_output_binary = []
        for sample_number in range(number_of_samples):
            # print(predicted_outputs)
            # print(predicted_outputs[sample_number][output_number])            
            if predicted_outputs[sample_number][0][output_number] < threshold:
                predicted_output = 0
            else:
                predicted_output = 1
            predicted_output_binary.append(predicted_output)

        print('accuracy: {0}'.format(sklearn.metrics.accuracy_score(
                                     true_outputs[:, output_number], 
                                     predicted_output_binary)))
        print(sklearn.metrics.confusion_matrix(true_outputs[:, output_number], 
                                               predicted_output_binary))


def training(model_iter):
    '''
    Performs Training of the specified network and outputs PNG images
    showing resulting learned weights and histograms of weights
    '''
    # Set parameters
    solver_prototxt_filename = 'toycar_solver.prototxt'
    train_test_prototxt_filename = 'toycar_2fc_hdf5.prototxt'
    caffemodel_filename = '2fc_iter_' + str(model_iter) + '.caffemodel' 

    # Train network
    train(solver_prototxt_filename)

    # Print network
    print_network(train_test_prototxt_filename, caffemodel_filename)
    print_network_weights(train_test_prototxt_filename, caffemodel_filename)


def testing(deploy_prototxt_filename, caffemodel_filename, inputs, labels):
    '''
    Performs Testing of the specified network
    '''    
    # Compute performance metrics
    outputs = get_predicted_outputs(deploy_prototxt_filename, 
                                    caffemodel_filename, inputs)
    
    print 'predictions: '
    print outputs[0]
    
    print 'ground truths: '
    print labels
    
    return outputs[0]
    
    
def euclidean_loss(pred, labels):
    '''
    Hand Calculate the Euclidean Loss to Compare with Model Output
    '''
    result = labels-pred
    size = pred.shape[0]
    
    loss = np.sum(np.square(result), axis=1)
    loss = np.sum(loss, axis=0) / (2 * size)
    
    print "Euclidean Loss: ", loss


if __name__ == "__main__":

    if sys.argv[1].lower() == "train":
        train_data, train_labels, test_data, test_labels = load_data()

        # save_data_as_hdf5('toycar_hdf5_data_random_norm11_train.hdf5', 
        #                   train_data, train_labels)
        # save_data_as_hdf5('toycar_hdf5_data_random_norm11_test.hdf5', 
        #                   test_data, test_labels)
        
        solver_name = "toycar_solver.prototxt"
        training(20000)

    elif sys.argv[1].lower() == "test":
        
        train_data, train_labels, test_data, test_labels = load_data()

        pred = testing('toycar_2fc_deploy.prototxt', 
                       '2fc_iter_100000.caffemodel', 
                       test_data[:1000, :], test_labels[:1000, :])

        pred[:, 0] = pred[:, 0] + test_data[:1000, 0]
        pred[:, 1] = pred[:, 1] + test_data[:1000, 1]

        euclidean_loss(pred, test_labels[:1000, :])
    else:
        train_data, train_labels, test_data, test_labels = load_data()


    # result = pred - test_labels[:1000, :]
    # print np.sqrt(np.sum(np.square(result), axis=1)).shape
