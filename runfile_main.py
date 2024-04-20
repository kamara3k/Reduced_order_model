#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:26:12 2024


"""

#%%
"""
Modules
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

#%% Load data

filename='drumsout_pinpowers_True_a11_2023-01-29.pkl'
with open(filename, "rb") as f:
    data_drumsout = pickle.load(f)
    
filename='drumsin_pinpowers_True_a11_2023-01-29.pkl'
with open(filename, "rb") as f:
    data_drumsin = pickle.load(f)

#%%
# check files are the same length
NN = len(data_drumsout)
if (NN!= len(data_drumsin)):
    import sys
    print('Data files not the same size!')
    sys.exit()
    
# check keys are the same
for k1,k2 in zip(data_drumsout.keys(),data_drumsin.keys()):
    if (k1!=k2):
        print(f'keys not the same! {k1}, {k2}')
        
#%% Construct data matrix X

X = np.empty((NN,2))

for n,key in enumerate(data_drumsout.keys()):
    #print(n,key)
    X[n,0] = data_drumsout[key]
    X[n,1] = data_drumsin[key]
    
fig, ax = plt.subplots()
ax.plot(X[:,0],'bx')
ax.plot(X[:,1],'g.') # I think it is suppoosed to be 1 not zero

#%% Perform POD

def proper_orthogonal_decomposition(data):
    # Subtract the mean from the data
    mean_data = np.mean(data, axis=0)
    data_centered = data - mean_data
    
    print(data_centered.shape)

    # Compute the covariance matrix
    covariance_matrix = np.cov(data_centered.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the POD modes
    #modes = np.dot(data_centered.T, eigenvectors)

    # Normalize modes
    #modes /= np.linalg.norm(modes, axis=0)

    return eigenvectors, eigenvalues, mean_data, covariance_matrix

    
eigenvectors, eigenvalues, X_mean, X_cov = proper_orthogonal_decomposition(X)

#%%

fig, ax = plt.subplots()
ax.plot(eigenvalues,'x')
# The verification of the SVD 
def POD_SVD(data):
    U, S, Vh = np.linalg.svd(data, full_matrices=True)
    n_comp=data.shape[1]
    U_r=U[:, :n_comp]
    S_r=S[:n_comp]
    Vh_r=Vh[:, :n_comp]
    return U_r, S_r,Vh_r, np.matmul(U_r * S_r, Vh_r) #previous return 
    #return U_r, S_r,Vh_r, np.matmul(U_r * S_r[:, np.newaxis], Vh_r)# becasue 
#1.'U_r * S_r' attempts to multiply  each column of  U_r by the vector S_r 
# which needs to be properly shaped as a 2D array for correct matrix multiplication
# 2. 'S_r[:, np.newaxis]' turns the 1D array Sr into a 2D array (column vector), 
# allowing for element-wise multiplication across the rows of Ur

U_r, S_r, Vh_r, X_tilda = POD_SVD(X)

error=X-X_tilda
#%%
def interpolate_pod(U_r, S_r, Vh_r, alpha=0.5):
    # Interpolate in the reduced space
    Vh_mid = Vh_r[0, :] * (1 - alpha) + Vh_r[1, :] * alpha

    # Reconstruct the mid position in the original space
    mid_position = U_r @ np.diag(S_r) @ Vh_mid
    return mid_position

# Call the function with alpha = 0.5 for midpoint
mid_position = interpolate_pod(U_r, S_r, Vh_r, alpha=0.5)
print("Interpolated Mid Position:\n", mid_position)
#Interpolation in Reduced Space: The interpolation is done by averaging the 
#right singular vectors (Vh_r) associated with the outside and inside positions.
#The weight alpha determines the interpolation factor, with 0.5 typically 
#representing the midpoint.
#Reconstruction: The interpolated lower-dimensional data ('Vh_mid') 
#is then used to reconstruct the data in the original high-dimensional space 
#using the matrices Ur and Σr obtained from SVD. 
#The product U_r @ np.diag(S_r) @ Vh_mid uses the reduced 
#left singular vectors and singular values to project the interpolated point 
#back to the original data space.
   
    
    
    