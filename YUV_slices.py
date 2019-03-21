# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:59:33 2018

Generate and show slices of YUV

@author: LocalAdmin
"""

import cv2;
import matplotlib.pyplot as plt
import numpy as np

def YUV_to_RGB(im):
    """ Convert YUV to RGB """
    if(np.max(im[:]) <= 1.0):
        im *= 255;
        
    Y = im[:,:,0];
    U = im[:,:,1];
    V = im[:,:,2];
    
    R  = Y + 1.402   * ( V - 128 )
    G  = Y - 0.34414 * ( U - 128 ) - 0.71414 * ( V - 128 )
    B  = Y + 1.772   * ( U - 128 )

    rgb = im;
    rgb[:,:,0] = R / 255.0;
    rgb[:,:,1] = G / 255.0;
    rgb[:,:,2] = B / 255.0;

    inds1 = np.where(rgb < 0.0);
    for i in range(len(inds1[0])):
        rgb[inds1[0][i], inds1[1][i], inds1[2][i]] = 0.0;
        
    inds2 = np.where(rgb > 1.0);
    for i in range(len(inds2[0])):
        rgb[inds2[0][i], inds2[1][i], inds2[2][i]] = 1.0;
    return rgb;

def generate_slices_YUV(n_slices = 5, H = 255, W = 255):
    """ Generate YUV slices and convert them to RGB to show them """
#    bgr = np.zeros([H, W, 3]);
#    bgr[:,:,2] = 1.0;
#    cv2.imshow('BGR', bgr);
#    cv2.waitKey();
    
    plt.ion();
    
    V_step = 1.0 / H;
    U_step = 1.0 / W;    
    Y_step = 1.0 / (n_slices-1);
    
    for s in range(n_slices):
        
        im = np.zeros([H, W, 3]);
        Y = s * Y_step;
        im[:,:,0] = Y;
        
        # print('Slice %d / %d, Y = %f' % (s, n_slices, Y));
        
        for y in range(H):
            for x in range(W):
                im[y,x,1] = U_step * x;
                im[y,x,2] = V_step * y;
                
        rgb = YUV_to_RGB(im);
        plt.figure()
        plt.imshow(rgb);
        plt.xlabel('U');
        plt.ylabel('V');
        plt.title('Y = ' + str(Y));

#        im=im.astype(np.float32)
#        bgr = cv2.cvtColor(im, cv2.COLOR_YUV2BGR);
#        cv2.imshow('BGR', bgr); cv2.waitKey();
    

def filter_color(image_name = 'DelFly_tulip.jpg', y_low = 50, y_high = 200, \
                 u_low = 120, u_high = 130, v_low = 120, v_high = 130, resize_factor=1):
    im = cv2.imread(image_name);
    im = cv2.resize(im, (int(im.shape[1]/resize_factor), int(im.shape[0]/resize_factor)));
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV);
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1]]);
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if(YUV[y,x,0] >= y_low and YUV[y,x,0] <= y_high and \
               YUV[y,x,1] >= u_low and YUV[y,x,1] <= u_high and \
               YUV[y,x,2] >= v_low and YUV[y,x,2] <= v_high):
                Filtered[y,x] = 1;
    
    plt.figure();
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
    plt.imshow(RGB);
    plt.title('Original image');
    
    plt.figure()
    plt.imshow(Filtered);
    plt.title('Filtered image');
    

if __name__ == '__main__':
    generate_slices_YUV();
    filter_color();