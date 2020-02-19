# Pasan Bandara - UofM - 7882322
# Assingment 2 Part 6
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

def compute_class(X,W,Y,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda):
    bottom = np.dot(X,W)
    bottom_e = np.exp(bottom)
    if np.any(np.isfinite(bottom_e)) and np.any(np.isin(0,bottom_e)) == 0: #Check exp saturation
        bottom_sum = np.sum(bottom_e,axis=1).reshape(no_of_training_examples,1)
        bottom_sum = np.repeat(bottom_sum, no_of_classes, axis=1)
        div = bottom_e/bottom_sum
        highest = np.argmax(div, axis=1).reshape(no_of_training_examples,1)
        correct = 0
        incorrect = 0
        confusion_matrix = np.zeros((10,10))
        for img in range(no_of_training_examples):
            confusion_matrix[Y[img,0],highest[img,0]] = confusion_matrix[Y[img,0],highest[img,0]] + 1
        #     if Y[img,0] == highest[img,0]:
        #         correct = correct +1
        #     else:
        #         incorrect = incorrect + 1
        # accuracy = (correct/no_of_training_examples)*100
        # print("Accuracy ",accuracy)
        return confusion_matrix
    else:
        print("Saturation Detected! Aborted! Saved so far learned data!")
        np.savetxt("Trained_Weights_Halfway.csv", W, delimiter=",")
        print("Exiting...")
        exit()


def main():
    lemda = 0.01
    alpha = 0.01
    epochs = 1
    no_of_training_examples = 10000
    no_of_input_features_with_b = 1025 #(with 1s for bias)
    no_of_classes = 10
    file_path = '/home/pasan/Documents/PythonCode/cifar-10-python/cifar-10-batches-py'
    data_path = '/home/pasan/Documents/PythonCode/AS2/Trained_Weights.csv'
    trained_weights = np.genfromtxt(data_path, delimiter=',')
    # airplane : 0 automobile : 1 bird : 2 cat : 3 deer : 4 dog : 5 frog : 6 horse : 7 ship : 8 truck : 9
    classes = np.array([0,1,2,3,4,5,6,7,8,9])
    batch_number = [2]

    W = trained_weights


    for batch_id in batch_number:
        print("Serving Now :  Batch ID : "+ str(batch_id))
        batch_features,batch_class = unpickle(file_path,batch_id)
        Gray = rgb2gray(batch_features)
        X_mean = np.mean(Gray)
        X_std_div = np.std(Gray)
        X_ori = (Gray - X_mean)/X_std_div

        temp2 = np.ones((no_of_training_examples,1))
        X = np.concatenate((temp2, X_ori), axis=1)

        Y = np.matrix(batch_class).transpose()

        conf_mat = compute_class(X,W,Y,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda)
        print("Confusion Matrix:")
        print(conf_mat)


if __name__== "__main__":
    main()
