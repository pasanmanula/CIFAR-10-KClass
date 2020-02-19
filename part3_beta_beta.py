# Pasan Bandara - UofM - 7882322
# Assingment 2 Part 3
from __future__ import division
import numpy as np
import cPickle
from sys import exit

def unpickle(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = cPickle.load(file)

    features = batch['data']
    labels = batch['labels']
    return features, labels

def rgb2gray(rgb_img):
    red_channel = rgb_img[:,0:1024]
    green_channel = rgb_img[:,1024:2048]
    blue_channel = rgb_img[:,2048:3072]
    gray_img =np.dot(0.2125,red_channel) + np.dot(0.7154,green_channel) + np.dot(0.0721,blue_channel)
    return gray_img

def compute_gradient(X,Y,Y_decompose,W,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda):
    bottom = np.dot(X,W)
    bottom_e = np.exp(bottom)
    if np.any(np.isfinite(bottom_e)) and np.any(np.isin(0,bottom_e)) == 0: #Check exp saturation
        bottom_sum = np.sum(bottom_e,axis=1).reshape(no_of_training_examples,1)
        bottom_sum = np.repeat(bottom_sum, no_of_classes, axis=1)
        div = bottom_e/bottom_sum
        diff = (Y_decompose - div)
        Mux = np.dot(X.transpose(),diff)
        scalar = -1/no_of_training_examples
        W_forced = np.copy(W)
        W_forced[0,:] = 0 #Remove bias from the weight matrix to calculate the regularization term
        final = scalar*Mux + lemda*W_forced
        return final
    else:
        print("Saturation Detected! Aborted! Saved so far learned data!")
        np.savetxt("Trained_Weights_Halfway.csv", W, delimiter=",")
        print("Exiting...")
        exit()


def main():
    lemda = 0.01
    alpha = 0.01
    epochs = 10000
    no_of_training_examples = 10000
    no_of_input_features_with_b = 1025 #(with 1s for bias)
    no_of_classes = 10
    file_path = '/home/pasan/Documents/PythonCode/cifar-10-python/cifar-10-batches-py'
    # airplane : 0 automobile : 1 bird : 2 cat : 3 deer : 4 dog : 5 frog : 6 horse : 7 ship : 8 truck : 9
    classes = np.array([0,1,2,3,4,5,6,7,8,9])
    batch_number = [1]

    W = np.random.randn(no_of_input_features_with_b,no_of_classes)*0.0001
    np.savetxt("Init_Weights.csv", W, delimiter=",")
    for epoch in range(epochs):
        for batch_id in batch_number:
            print("Serving Now : Epoch Number : "+str(epoch)+" Batch ID : "+ str(batch_id))
            batch_features,batch_class = unpickle(file_path,batch_id)
            Gray = rgb2gray(batch_features)
            X_mean = np.mean(Gray)
            X_std_div = np.std(Gray)
            X_ori = (Gray - X_mean)/X_std_div

            temp2 = np.ones((no_of_training_examples,1))
            X = np.concatenate((temp2, X_ori), axis=1)

            Y = np.matrix(batch_class).transpose()
            Y_decompose = np.zeros((no_of_training_examples,no_of_classes))
            for k in classes:
                for row in range(0,no_of_training_examples):
                    if Y[row,0] == k:
                        Y_decompose[row,k] = 1
                    else:
                        Y_decompose[row,k] = 0
            gradient_mat = compute_gradient(X,Y,Y_decompose,W,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda)
            W = W - alpha*gradient_mat
    print("Training has finished. Saving the weight matrix!")
    np.savetxt("Trained_Weights.csv", W, delimiter=",")
    print("Done!")

if __name__== "__main__":
    main()
