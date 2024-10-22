#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Homework code written by 
Vinay Lanka 
vlanka@umd.edu
M.Eng Robotics
120417665
"""

# Code starts here:

import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.transform import rotate
import sklearn.cluster
import matplotlib.pyplot as plt
import math

def image_loader(path):
    test = cv2.imread(path)
    return test
    # cv2.imshow("test", test)
    # cv2.waitKey(0)

def gaussian_2d(kernel_size,sigma):
  sigma_x = sigma[0]
  sigma_y = sigma[1]
  x, y = np.meshgrid(np.linspace(-kernel_size/2, kernel_size/2, kernel_size),
                      np.linspace(-kernel_size/2, kernel_size/2, kernel_size))
  norm = (1/np.sqrt(2 * np.pi * (sigma_x * sigma_y)))
  pow = -1 * (((x**2)/(sigma_x**2)) + ((y**2)/(sigma_y**2)))
  result = norm * np.exp(pow)
  return result

def DoG_generator(filter_size=19, sigma=1, scales=2, orientations=16):
    #Gaussian Kernel
    #Default Base filter size = 5, Scales s = 2, Orientations o = 16, Sigma = 1
    gaussian_filters = []
    angles = np.linspace(0,360,num=orientations)
    for scale in range(0,scales):
        for orientation in range(0,orientations):
            G = gaussian_2d(filter_size,[(scale + 1)*sigma,(scale + 1)*sigma])
            sobel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
            DoG = convolve2d(G,sobel,boundary='fill', mode='valid')
            oriented_DoG = rotate(DoG,angles[orientation])
            gaussian_filters.append(oriented_DoG)
    return gaussian_filters

def LM_generator(filter_size, scales, orientations):
    LM_filters = []

    #Gaussian
    for scale in scales:
        G = gaussian_2d(filter_size,[scale, scale])
        LM_filters.append(G)
        # plt.imshow(G,cmap='gray')
        # plt.show()
    #LoG
    log_scale = scales + [scale * 3 for scale in scales]
    for scale in log_scale:
        G = gaussian_2d(filter_size,[scale, scale])
        laplacian = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=np.float32)
        LoG = convolve2d(G,laplacian,boundary='fill', mode='valid')
        LM_filters.append(LoG)
        # plt.imshow(LoG,cmap='gray')
        # plt.show()
    #First and Second Derivative with Elongation
    angles = np.linspace(0,360,num=orientations)
    for scale in scales[0:3]:
        for angle in angles:
            G = gaussian_2d(filter_size,[scale, 3 * scale])
            sobel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
            first_derivative = convolve2d(G,sobel,boundary='fill', mode='valid')
            second_derivative = convolve2d(first_derivative,sobel,boundary='fill', mode='valid')
            oriented_first_derivative = rotate(first_derivative,angle)
            oriented_second_derivative = rotate(second_derivative,angle)
            LM_filters.append(oriented_first_derivative)
            LM_filters.append(oriented_second_derivative)
            # plt.imshow(oriented_second_derivative,cmap='gray')
            # plt.show()
    return LM_filters

def Gabor_generator(filter_size, scales, orientations, frequencies):
    Gabor_filters = []
    for scale in scales:
        G = gaussian_2d(filter_size, [scale,scale])
        for frequency in frequencies:
            angles = np.linspace(0,360,num=orientations)
            for angle in angles:
                x, y = np.meshgrid(np.linspace(-filter_size/2, filter_size/2, filter_size),
                      np.linspace(-filter_size/2, filter_size/2, filter_size))
                u = x * np.cos(angle) + y * np.sin(angle)
                sinusoid = np.sin(2 * np.pi * u * frequency/filter_size)
                Gabor_filter = G * sinusoid
                Gabor_filters.append(Gabor_filter)
                # plt.imshow(Gabor_filter,cmap='gray')
                # plt.show()
    return Gabor_filters

def halfDisk(radius, angle):
	size = 2*radius + 1
	centre = radius
	half_disk = np.zeros([size, size])
	for i in range(radius):
		for j in range(size):
			distance = np.square(i-centre) + np.square(j-centre)
			if distance <= np.square(radius):
				half_disk[i,j] = 1
	
	half_disk = rotate(half_disk, angle)
	half_disk[half_disk<=0.5] = 0
	half_disk[half_disk>0.5] = 1
	return half_disk

def Half_disk_generator(scales,orientations):
    half_disk_masks = []
    for scale in scales:
        angles = [0,180,90,270,45,215,135,305]
        for angle in angles:
            half_disk_mask = halfDisk(scale, angle)
            half_disk_masks.append(half_disk_mask)

    return half_disk_masks

def save_filters(filter_bank, file_name):
    rows = math.ceil(len(filter_bank)/6)
    plt.subplots(rows, 6, figsize=(15,15))
    for idx, filter in enumerate(filter_bank):
        plt.subplot(rows, 6, idx+1)
        plt.axis('off')
        plt.imshow(filter, cmap='gray')
    plt.savefig(file_name)
    plt.close()

def apply_filters(image, filter_bank):
    filtered_images = []	
    for filter in filter_bank:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = convolve2d(image_gray,filter,boundary='fill', mode='same')
        filtered_images.append(filtered_image)
    return filtered_images

def chi_square_distance(image, bins, filter_bank):
    chi_square_distances = []
    N = len(filter_bank)
    n = 0
    while n < N:
        left_mask = filter_bank[n]
        right_mask = filter_bank[n+1]		
        tmp = np.zeros(image.shape)
        chi_sq_dist = np.zeros(image.shape)
        min_bin = np.min(image)
        for bin in range(bins):
            tmp[image == bin+min_bin] = 1
            # g_i = cv2.filter2D(tmp,-1,left_mask)
            # h_i = cv2.filter2D(tmp,-1,right_mask)
            g_i = convolve2d(tmp,left_mask,boundary='fill',mode='same')
            h_i = convolve2d(tmp,right_mask,boundary='fill',mode='same')
            chi_sq_dist += (g_i - h_i)**2/(g_i + h_i + np.exp(-7))
        chi_sq_dist /= 2
        chi_square_distances.append(chi_sq_dist)
        n = n+2
    return chi_square_distances

def get_edges(T_g, B_g, C_g, canny_edges, sobel_edges, weights):
	canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_BGR2GRAY)
	sobel_edges = cv2.cvtColor(sobel_edges, cv2.COLOR_BGR2GRAY)
	T1 = (T_g + B_g + C_g)/3
	T2 = (weights[0] * canny_edges) + (weights[1] * sobel_edges)
	pb_lite_op = np.multiply(T1, T2)
	return pb_lite_op

def main():
    folder_name = "./"
    """
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
    DoG_filters = DoG_generator(filter_size=19,sigma=2,scales=2, orientations=16)
    save_filters(DoG_filters,folder_name + "results/Filters/DoG.png")
    """
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
    LM_Small_filters = LM_generator(filter_size=27,scales=[1, np.sqrt(2), 2, 2*np.sqrt(2)],orientations=6)
    LM_Large_filters = LM_generator(filter_size=27,scales=[np.sqrt(2), 2, 2*np.sqrt(2),4],orientations=6)
    save_filters(LM_Small_filters,folder_name + "results/Filters/LM_Small.png")
    save_filters(LM_Large_filters,folder_name + "results/Filters/LM_Large.png")
    """
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
    Gabor_filters = Gabor_generator(filter_size=27,scales=[5,10],orientations=6, frequencies=[4,6,8])
    save_filters(Gabor_filters,folder_name + "results/Filters/Gabor.png")
    """
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
    Half_disk_masks = Half_disk_generator(scales=[2,4,6,8,10],orientations=6)
    save_filters(Half_disk_masks,folder_name + "results/Filters/HalfDisk.png")
    """
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
    filter_bank = DoG_filters + LM_Large_filters + LM_Small_filters + Gabor_filters
    for i in range(1,11):
        image = image_loader("./BSDS500/Images/" + str(i) + ".jpg")
        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """
        texture_bins = 64
        filtered_image = np.array(apply_filters(image, filter_bank))
        # plt.imsave("./results/FilteredImage/"+str(i)+".png",filtered_image)

        f,x,y = filtered_image.shape
        image_flattened = filtered_image.reshape([f, x*y]).transpose()
        kmeans = sklearn.cluster.KMeans(n_clusters = texture_bins, n_init=4).fit(image_flattened)
        labels = kmeans.labels_
        # labels = kmeans.predict(image_flattened)
        texton_image = labels.reshape([x,y])
        plt.imsave("./results/TextonMap/"+str(i)+".png",texton_image)

        T_g = np.mean(np.array(chi_square_distance(texton_image, texture_bins, Half_disk_masks)),axis=0)
        plt.imsave("./results/T_g/"+str(i)+".png",T_g)

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        brightness_bins = 16
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x,y = image_gray.shape
        image_flattened = image_gray.reshape([x*y,1])
        kmeans = sklearn.cluster.KMeans(n_clusters = brightness_bins, random_state=4).fit(image_flattened)
        # labels = kmeans.predict(image_flattened)
        labels = kmeans.labels_
        brightness_image = labels.reshape([x,y])
        plt.imsave("./results/BrightnessMap/"+str(i)+".png",brightness_image)

        B_g = np.mean(np.array(chi_square_distance(brightness_image, brightness_bins, Half_disk_masks)),axis = 0)
        plt.imsave("./results/B_g/"+str(i)+".png",B_g)

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        color_bins = 16
        x,y,c = image.shape
        image_flattened = image.reshape([x*y,c])
        kmeans = sklearn.cluster.KMeans(n_clusters = color_bins, random_state=4).fit(image_flattened)
        # labels = kmeans.predict(image_flattened)
        labels = kmeans.labels_
        color_image = labels.reshape([x,y])
        plt.imsave("./results/ColorMap/"+str(i)+".png",color_image)

        C_g = np.mean(np.array(chi_square_distance(color_image, color_bins, Half_disk_masks)),axis=0)
        plt.imsave("./results/C_g/"+str(i)+".png",C_g)

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        canny_baseline = cv2.imread("./BSDS500/CannyBaseline/"+ str(i) +".png")
        sobel_baseline = cv2.imread("./BSDS500/SobelBaseline/"+ str(i) +".png")
        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pb_edge = get_edges(T_g, B_g, C_g, canny_baseline, sobel_baseline, [0.5,0.5])
        plt.imsave("./results/PbLite_results/PbLite"+str(i)+".png",pb_edge,cmap='gray')
        # plt.imshow(pb_edge, cmap = "gray")
        # plt.show()

if __name__ == '__main__':
    main()