# Pasan Bandara - UofM - 7882322
# Assingment 2 Part 1
from __future__ import division
import numpy as np
from sys import exit


def compute_loss(X,Y,Y_decompose,W,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda):
    bottom = np.dot(X,W)
    bottom_e = np.exp(bottom)
    if np.any(np.isfinite(bottom_e)) and np.any(np.isin(0,bottom_e)) == 0: #Check exp saturation
        bottom_sum = np.sum(bottom_e,axis=1).reshape(no_of_training_examples,1)
        bottom_sum = np.repeat(bottom_sum, no_of_classes, axis=1)
        div = np.log(bottom_e/bottom_sum)
        Mux = np.sum(np.sum(np.multiply(Y_decompose,div),axis=1))
        fin = -1*Mux/no_of_training_examples
        W_forced = np.copy(W)
        W_forced[0,:] = 0 #Remove bias from the weight matrix to calculate the regularization term
        reg = lemda/2*np.sum(np.sum(np.dot(W_forced.transpose(),W_forced),axis=1))
        final = fin+ reg
        return final
    else:
        print("Saturation Detected! Aborted! Saved so far learned data!")
        print("Exiting...")
        exit()

def main():
    lemda = 0.01
    no_of_training_examples = 2
    no_of_input_features_with_b = 4 #(with 1s for bias)
    no_of_classes = 2
    classes = np.array([0,1])
    batch_number = [1]

    W = np.array([[0.12,0.13],[-0.12,0.14],[-0.15,0.16],[-0.15,-0.16]])

    X_ori = np.array([[2,3,4],[5,6,7]])
    temp2 = np.ones((no_of_training_examples,1)) #1s for bias
    X = np.concatenate((temp2, X_ori), axis=1)
    Y = np.array([[1],[0]])
    Y_decompose = np.zeros((no_of_training_examples,no_of_classes))
    for k in classes:
        for row in range(0,no_of_training_examples):
            if Y[row,0] == k:
                Y_decompose[row,k] = 1
            else:
                Y_decompose[row,k] = 0
    loss = compute_loss(X,Y,Y_decompose,W,classes,no_of_training_examples,no_of_input_features_with_b,no_of_classes,lemda)
    print("Loss Value:",loss)



if __name__== "__main__":
    main()
