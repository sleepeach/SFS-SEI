import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset_ADS_B():
	X = np.load("E:\pythoncode\ADS-Bista-torch\ADS_B_4800\Task_1_Train_X_100Class.npy")
	X = X.transpose(2, 0, 1)
	Y = np.load("E:\pythoncode\ADS-Bista-torch\ADS_B_4800\Task_1_Train_Y_100Class.npy")
	Y =Y.astype(np.uint8)
	X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=30)
	return X_train, X_val, Y_train, Y_val

def TestDataset_ADS_B():
	X = np.load("E:\pythoncode\ADS-Bista-torch\ADS_B_4800\Task_1_Test_X_100Class.npy")
	X = X.transpose(2, 0, 1)
	Y = np.load("E:\pythoncode\ADS-Bista-torch\ADS_B_4800\Task_1_Test_Y_100Class.npy")
	Y = Y.astype(np.uint8)
	return X, Y

def getdata_ADS_B():
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset_ADS_B()
    X_test, Y_test = TestDataset_ADS_B()
    return X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test