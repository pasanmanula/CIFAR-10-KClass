# Pasan Bandara - UofM - 7882322
# Assingment 2 Part 2
from __future__ import division
import numpy as np
from sys import exit


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
        np.savetxt("Trained_Weights_Halfway_part2.csv", W, delimiter=",")
        print("Exiting...")
        exit()

def main():
    lemda = 0.01
    alpha = 0.01
    epochs = 10
    no_of_training_examples = 2
    no_of_input_features_with_b = 4 #(with 1s for bias)
    no_of_classes = 2
    classes = np.array([0,1])
    batch_number = [1]

    W = np.array([[0.12,0.13],[-0.12,0.14],[-0.15,0.16],[-0.15,-0.16]])
    np.savetxt("Init_Weights_part2.csv", W, delimiter=",")
    for epoch in range(epochs):
        for batch_id in batch_number:
            print("Serving Now : Epoch Number : "+str(epoch)+" Batch ID : "+ str(batch_id))

            X_ori = np.array([[2,3,4],[5,6,7]])
            temp2 = np.ones((no_of_training_examples,1))
            X = np.concatenate((temp2, X_ori), axis=1)
            Y = np.array([[1],[0]])
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
    np.savetxt("Trained_Weights_part2.csv", W, delimiter=",")
    print("Done!")

if __name__== "__main__":
    main()
