
# coding: utf-8

# Author:Sanjoy Basu
# 
# 

# 
# Data is an image of Tofukuji temple, in southeastern Kyoto,Japan. I (Sanjoy Basu) am the photographer of the image.
# 
# 




import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import cluster
import matplotlib.image as mpimg
from skimage.io import imread, imshow, imsave
import os




#Loading the data
kyoto=imread("kyoto.jpg")
#Displaying the image
imshow(kyoto)



# Making the image compatible with matplotlib 
# Matplotlib represent RGB from value 0 to 1
kyoto_f64=kyoto.astype(dtype=np.float64)/255
#Ensuring conversion did not significantly alter the image
imshow(kyoto_f64)



# Determining the shape
kyoto_f64.shape




kyoto_f64.shape[2]


# Unlike most dataset which is AxB matrix image is AxBxC matrix. In other word image is reprented by
# 
# (Height x Width x Pixels) matix. 



# Extracting pixel from data set
px_d=[] #Temporarily store the pixel after grabbing the pixel at a locaton
px_v=[] # Warehousing the pixels

for i in range(kyoto_f64.shape[0]):
    for j in range(kyoto_f64.shape[1]):
      
      
        for k in kyoto_f64[i, j,:]:
           
            px_d.append(k)
           
        px_v.append(px_d) #Storing the the pixel 
        px_d=[] #Empties the pixel to grab a new one
        
# Turning the data into an array
vArray=np.array(px_v)

# Turning the araray into feature vector

vArrayDF=pd.DataFrame(vArray, columns=['r', 'g', 'b'])
vArrayDF.head(10) #Displaying the pixels


# K-Means Clustering



km=cluster.KMeans(init='k-means++', n_clusters=64, n_init=10)
km.fit(vArrayDF)
ym=km.predict(vArrayDF)


#  Generating codebook



codebook=km.cluster_centers_


# Building function to recreate image 


def recreate_image(codebook, labels, y, x):
    #Creating all zero 3 dimension matrix. Information from codebook will be used to generate the pixels
    image = np.zeros((y, x, codebook.shape[1])) 
    label_idx = 0
    for i in range(y):
        for j in range(x):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Recreating the image


image=recreate_image(km.cluster_centers_, ym, kyoto_f64.shape[0], kyoto_f64.shape[1])
#Verifying the shape of image 
image.shape


# Shape of the new image is same as the original.So we will display the image
# 



imshow(image)


# New image looks similar to the original image without significant loss of information.
# We will now save the image onthe disk for validation. We will call new image kyoto_compressed.jpg



imsave('kyoto_compressed.jpg', image)


# Validation



origInal=os.path.getsize("kyoto.jpg")
compRessed=os.path.getsize("kyoto_compressed.jpg")
100*(origInal-compRessed)/origInal


# Conclusion
# 
# Compressed image size is 54% the size of original image
# 





