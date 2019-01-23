# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:59:33 2018

Generate and show slices of YUV

@author: LocalAdmin
"""

#import cv2;
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

def generate_slices_YUV(n_slices = 5, H = 400, W = 400):
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
        plt.title('Y = ' + str(Y));
        
#        im=im.astype(np.float32)
#        bgr = cv2.cvtColor(im, cv2.COLOR_YUV2BGR);
#        cv2.imshow('BGR', bgr); cv2.waitKey();
if __name__ == '__main__':
    generate_slices_YUV();