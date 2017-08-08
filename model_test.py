#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:21:38 2017

@author: kramerPro

Testing different probability models for the creation of MRI data
"""
from PIL import Image
from scipy import stats
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


# Reading in an image
im = mpimg.imread("./data/training_ss_151.png") 
im_seg = mpimg.imread("./data/training_seg_151.png")

im_lisa = mpimg.imread("./data/001_4.bmp") 
im_seg_lisa = mpimg.imread("./data/001_4_seg.bmp")
# Yellow is white matter
# green is csf
# blue is grey
# red is damaged \\ white (I think)

# getting one pixel
plt.imshow(im_seg_lisa[80:81, 50:51]) 
# Blue [1, 102, 255] grey matter (1)
plt.imshow(im_seg_lisa[85:86, 50:51]) 
# Yellow [241, 255, 1] white matter (2)
plt.imshow(im_seg_lisa[50:51, 80:81]) 
# Green array([[[  1, 255,  71]]], dtype=uint8) csf (3)
plt.imshow(im_seg_lisa[33:34, 70:71])
# Red array([[[255,   1,  35]]], dtype=uint8) disease (4)

#### set lisa to black if not segmented

#def get_segmented_lisa(image):
#    "turns everything black that is not segmented"
#    crds = [] # saves a tuple ((x,y),lable)
#    for i in range(len(image)):
#        for j in range(len(image[1])):
#            if tuple(image[i,j]) == (1, 102, 255):
#                crds.append((i,j,1)) 
#            elif tuple(image[i,j]) == (241, 255, 1):
#                crds.append((i,j,2))
#            elif tuple(image[i,j]) == (1, 255, 71):
#                crds.append((i,j,3))
#            elif tuple(image[i,j]) == (255, 1, 35):
#                crds.append((i,j,4))
#            else:
#                image[i,j] = 1
#    coords = np.empty((len(crds),3))
#    for i in range(len(crds)):
#        coords[i] = crds[i]
#    return coords.astype(int)
#
#
#im_test_coords = get_segmented_lisa(im_test)

##### test #####
def get_image_data(image, seg_image):
    '''
    Turns everything black that is not segmented.
    Works for lisa's data
    
    The image is greyscale so channels are equal
        We only need 1 number to represent all of it
    
    '''
    crds = [] # saves a tuple ((x,y),lable)
    neighbors = [] # saves pixel value of neighbors for EM
    for i in range(len(seg_image)):
        for j in range(len(seg_image[1])):
            if tuple(seg_image[i,j]) == (1, 102, 255):
                crds.append((i,j,image[i,j][0],1)) # one value grayscale
                # grey matter
                neighbors.append((im_lisa[(i-3):(i+2),(j-3):(j+2),1].reshape((1,25)),1))
                # [sub for each in im[(i-3):(i+2),(j-3):(j+2),1] for sub in each]
            elif tuple(seg_image[i,j]) == (241, 255, 1):
                crds.append((i,j,image[i,j][0],2))
                # white
                neighbors.append((im_lisa[(i-3):(i+2),(j-3):(j+2),1].reshape((1,25)),2))
            elif tuple(seg_image[i,j]) == (1, 255, 71):
                crds.append((i,j,image[i,j][0],3))
                # csf
                neighbors.append((im_lisa[(i-3):(i+2),(j-3):(j+2),1].reshape((1,25)),3))
            elif tuple(seg_image[i,j]) == (255, 1, 35):
                crds.append((i,j,image[i,j][0],4))
                # disease
                neighbors.append((im_lisa[(i-3):(i+2),(j-3):(j+2),1].reshape((1,25)),4))
            else:
                seg_image[i,j] = 1
                image[i,j] = 1
    coords = np.empty((len(crds),4))
    neighbors_data = np.empty((len(crds),26))
    for i in range(len(crds)):
        coords[i] = crds[i]
        neighbors_data[i] = np.append(neighbors[i][0],neighbors[i][1])
    return coords.astype(int), neighbors_data

data_test, neighbors_d = get_image_data(im_lisa, im_seg_lisa)

# used https://stackoverflow.com/questions/10377998/



def get_all_data():
    directory = os.fsencode("./data")
    image_names = []
    seg_image_names = []
    images = []
    images_seg = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".bmp"): 
            if filename.endswith("seg.bmp"): 
                seg_image_names.append(filename)
            else:
                image_names.append(filename)
        else:
            continue
    images.append(mpimg.imread("./data/"+image_names[0]))
    images_seg.append(mpimg.imread("./data/"+seg_image_names[0]))
    data_coord, data_neigh = get_image_data(images[0], images_seg[0])
    for i in range(1,len(image_names)):
        images.append(mpimg.imread("./data/"+image_names[i])) 
        images_seg.append(mpimg.imread("./data/"+seg_image_names[i]))
        data_coord1, data_neigh1 = get_image_data(images[i], images_seg[i])
        data_coord = np.append(data_coord, data_coord1,0)
        data_neigh = np.append(data_neigh, data_neigh1,0)
    return data_coord, data_neigh, images, images_seg

data_coord, data_neigh, images, images_seg = get_all_data()

data = data_coord
####### kmeans #########













##########################
grey_cluster = data[data[:,3]==1]

plt.scatter(grey_cluster[:,0],grey_cluster[:,1])

plt.hist2d(grey_cluster[:,0],grey_cluster[:,1], bins=20)
plt.colorbar()
plt.title("grey matter cluster histogram")
plt.show()

plt.hist(grey_cluster[:,2])


grey_cluster_mu = np.mean(grey_cluster[:,2])
grey_cluster_var = np.var(grey_cluster[:,2])

white_cluster = data[data[:,3]==2]

plt.scatter(white_cluster[:,0],white_cluster[:,1])

plt.hist2d(white_cluster[:,0],white_cluster[:,1], bins=20)
plt.colorbar()
plt.title("white matter cluster histogram")
plt.show()

plt.hist(white_cluster[:,2])

white_cluster_mu = np.mean(white_cluster[:,2])
white_cluster_var = np.var(white_cluster[:,2])

csf_cluster = data[data[:,3]==3]

plt.scatter(csf_cluster[:,0],csf_cluster[:,1])

plt.hist2d(csf_cluster[:,0],csf_cluster[:,1], bins=20)
plt.colorbar()
plt.title("csf cluster histogram")
plt.show()

plt.hist(csf_cluster[:,2])

csf_cluster_mu = np.mean(csf_cluster[:,2])
csf_cluster_var = np.var(csf_cluster[:,2])

disease_cluster = data[data[:,3]==4]

plt.scatter(disease_cluster[:,0],disease_cluster[:,1])

plt.hist2d(disease_cluster[:,0],disease_cluster[:,1], bins=20)
plt.colorbar()
plt.title("disease cluster histogram")
plt.show()

plt.hist(disease_cluster[:,2])

disease_cluster_mu = np.mean(disease_cluster[:,2])
disease_cluster_var = np.var(disease_cluster[:,2])

################ With Symmetry
# I'm going to use symmetry to center and reflect the data
reflection  = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
test = data - [0,np.mean(data[:,1]),0,0]

test[:,1] = np.absolute(test[:,1])

grey_cluster = test[test[:,3]==1]

plt.scatter(grey_cluster[:,0],grey_cluster[:,1])

plt.hist2d(grey_cluster[:,0],grey_cluster[:,1], bins=20)
plt.colorbar()
plt.title("grey matter cluster histogram")
plt.show()

plt.hist(grey_cluster[:,2])


### going to generate some data with the my model
N = 20000
K = 4
pi = [len(grey_cluster)/len(data),len(white_cluster)/len(data),
      len(csf_cluster)/len(data),len(disease_cluster)/len(data)]
means = [grey_cluster_mu, white_cluster_mu, csf_cluster_mu, disease_cluster_mu]
sigs = [grey_cluster_var, white_cluster_var, csf_cluster_var, disease_cluster_var]




#z_inter = np.zeros((N,K),dtype=int)
#for i in range(len(z)):
#    z_inter[i][z[i]]=1
#z = z_inter

muX1 = np.mean(grey_cluster,0)[0:2]
muX2 = np.mean(white_cluster,0)[0:2]
muX3 = np.mean(csf_cluster,0)[0:2]
muX4 = np.mean(disease_cluster,0)[0:2]

covX1 = np.cov(grey_cluster[:,0:2].T)
covX2 = np.cov(white_cluster[:,0:2].T)
covX3 = np.cov(csf_cluster[:,0:2].T)
covX4 = np.cov(disease_cluster[:,0:2].T)

mus = [muX1, muX2, muX3, muX4]
covs = [covX1, covX2, covX3, covX4]

z = np.random.choice(range(4),size=N,p=pi)

sim_data = np.zeros((N,3))

for i in range(N):
    sim_data[i,0:2] = np.random.multivariate_normal(mus[z[i]],covs[z[i]] ,size=1)
    if sim_data[i,0] > 127:
        sim_data[i,0] = 127
    elif sim_data[i,1] > 191:
        sim_data[i,1] = 191
    sim_data[i,2] = np.random.normal(means[z[i]],np.sqrt(sigs[z[i]]),1)


sim_data = np.round(sim_data).astype(int)

im_test = im_lisa
im_test[:,:]=1
plt.imshow(im_test)
for i in range(N):
    im_test[sim_data[i][0], sim_data[i][1]] = sim_data[i][2]

plt.imshow(im_test)


################ Do this with reflection #############

data = data - [0,np.mean(data[:,1]),0,0]

data[:,1] = np.absolute(test[:,1])

grey_cluster = data[data[:,3]==1]

plt.scatter(grey_cluster[:,0],grey_cluster[:,1])

plt.hist2d(grey_cluster[:,0],grey_cluster[:,1], bins=20)
plt.colorbar()
plt.title("grey matter cluster histogram")
plt.show()

plt.hist(grey_cluster[:,2])


grey_cluster_mu = np.mean(grey_cluster[:,2])
grey_cluster_var = np.var(grey_cluster[:,2])

white_cluster = data[data[:,3]==2]

plt.scatter(white_cluster[:,0],white_cluster[:,1])

plt.hist2d(white_cluster[:,0],white_cluster[:,1], bins=20)
plt.colorbar()
plt.title("white matter cluster histogram")
plt.show()

plt.hist(white_cluster[:,2])

white_cluster_mu = np.mean(white_cluster[:,2])
white_cluster_var = np.var(white_cluster[:,2])

csf_cluster = data[data[:,3]==3]

plt.scatter(csf_cluster[:,0],csf_cluster[:,1])

plt.hist2d(csf_cluster[:,0],csf_cluster[:,1], bins=20)
plt.colorbar()
plt.title("csf cluster histogram")
plt.show()

plt.hist(csf_cluster[:,2])

csf_cluster_mu = np.mean(csf_cluster[:,2])
csf_cluster_var = np.var(csf_cluster[:,2])

disease_cluster = data[data[:,3]==4]

plt.scatter(disease_cluster[:,0],disease_cluster[:,1])

plt.hist2d(disease_cluster[:,0],disease_cluster[:,1], bins=20)
plt.colorbar()
plt.title("disease cluster histogram")
plt.show()

plt.hist(disease_cluster[:,2])

disease_cluster_mu = np.mean(disease_cluster[:,2])
disease_cluster_var = np.var(disease_cluster[:,2])

################ With Symmetry
# I'm going to use symmetry to center and reflect the data
#reflection  = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
#test = data - [0,np.mean(data[:,1]),0,0]
#
#test[:,1] = np.absolute(test[:,1])

grey_cluster = test[test[:,3]==1]

plt.scatter(grey_cluster[:,0],grey_cluster[:,1])

plt.hist2d(grey_cluster[:,0],grey_cluster[:,1], bins=20)
plt.colorbar()
plt.title("grey matter cluster histogram")
plt.show()

plt.hist(grey_cluster[:,2])


### going to generate some data with the my model
N = 20000
K = 4
pi = [len(grey_cluster)/len(data),len(white_cluster)/len(data),
      len(csf_cluster)/len(data),len(disease_cluster)/len(data)]
means = [grey_cluster_mu, white_cluster_mu, csf_cluster_mu, disease_cluster_mu]
sigs = [grey_cluster_var, white_cluster_var, csf_cluster_var, disease_cluster_var]




#z_inter = np.zeros((N,K),dtype=int)
#for i in range(len(z)):
#    z_inter[i][z[i]]=1
#z = z_inter

muX1 = np.mean(grey_cluster,0)[0:2]
muX2 = np.mean(white_cluster,0)[0:2]
muX3 = np.mean(csf_cluster,0)[0:2]
muX4 = np.mean(disease_cluster,0)[0:2]

covX1 = np.cov(grey_cluster[:,0:2].T)
covX2 = np.cov(white_cluster[:,0:2].T)
covX3 = np.cov(csf_cluster[:,0:2].T)
covX4 = np.cov(disease_cluster[:,0:2].T)

mus = [muX1, muX2, muX3, muX4]
covs = [covX1, covX2, covX3, covX4]

z = np.random.choice(range(4),size=N,p=pi)

sim_data = np.zeros((N,3))


# you still need to handle the reflection
for i in range(N):
    sim_data[i,0:2] = np.random.multivariate_normal(mus[z[i]],covs[z[i]] ,size=1)
    if sim_data[i,0] > 127:
        sim_data[i,0] = 127
    elif sim_data[i,1] > 191:
        sim_data[i,1] = 191
    sim_data[i,2] = np.random.normal(means[z[i]],np.sqrt(sigs[z[i]]),1)


sim_data = np.round(sim_data).astype(int)

im_test = im_lisa
im_test[:,:]=1
plt.imshow(im_test)
for i in range(N):
    im_test[sim_data[i][0], sim_data[i][1]] = sim_data[i][2]

plt.imshow(im_test)
