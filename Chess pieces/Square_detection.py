import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

img_path = 'D:/github aicrowd/Chess pieces/23.jpg'

image = cv2.imread(img_path)
cv2.imshow('image',image)




def skeletonize(blur):
    arr1 = blur.copy()
    for i in range(0,arr1.size-1):
        if blur[i] <= arr1[i+1]:
            arr1[i] = 0

    for i in np.arange(arr1.size-1,0,-1):
        if arr1[i-1] > blur[i]:
            arr1[i] = 0
    return arr1


gray_img1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#gray_img1= cv2.GaussianBlur(gray_img1,(5,5),0)
#gray_img1 = cv2.medianBlur(gray_img1,5)
gray_img1 = cv2.bilateralFilter(gray_img1,9,75,75)

sobel_x = cv2.Sobel(gray_img1,cv2.CV_64F,1,0,ksize=3)
cv2.imshow('sobel_x',sobel_x)

sobel_y = cv2.Sobel(gray_img1,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('sobel_y',sobel_y)

sobel_y_clipped_pos = np.clip(sobel_y,0.,255.)
sobel_y_clipped_neg = np.clip(sobel_y,-255.,0)
sum_y_pos = np.sum(sobel_y_clipped_pos,axis=1)
#plt.plot(sum_y_pos)
#plt.show()

sum_y_neg = np.sum(sobel_y_clipped_neg,axis=1)
#plt.plot(sum_y_neg)
#plt.show()

combined_sum_y = (sum_y_pos * (-1*sum_y_neg))/(254*254)
plt.plot(combined_sum_y)
plt.show()
#Dx_pos = np.clip(sobel_x,0.,255.)
#Dx_neg = np.clip(sobel_x,-255.,0.)


sobel_x_clipped_pos = np.clip(sobel_x,0.,255.)
sobel_x_clipped_neg = np.clip(sobel_x,-255.,0)
sum_x_pos = np.sum(sobel_x_clipped_pos,axis=0)
#plt.plot(sum_x_pos)
#plt.show()

sum_x_neg = np.sum(sobel_x_clipped_neg,axis=0)
#plt.plot(sum_x_neg)
#plt.show()

combined_sum_x = (sum_x_pos * (-1*sum_x_neg))/(254*254)
plt.plot(combined_sum_x)
plt.show()

hough_Dx_thresh = np.max(combined_sum_x)*3/5
hough_Dy_thresh = np.max(combined_sum_y)*3/5

#
blur_x = combined_sum_x > hough_Dx_thresh
blur_y = combined_sum_y > hough_Dy_thresh

#plt.plot(blur_x)
#plt.show()

skel_x = skeletonize(blur_x)
skel_y = skeletonize(blur_y)

plt.plot(skel_x)
plt.show()

lines_x = np.where(skel_x)[0]
lines_y = np.where(skel_y)[0]

is_match = len(lines_x) ==7 and len(lines_y)==7

plt.imshow(image)

for hx in lines_x:
    plt.axvline(hx,color='b',lw=2)
for hy in lines_y:
    plt.axhline(hy,color='r',lw=2)
#cv2.imshow('grayIimg',gray_img1)


step_x = np.int32(np.round(np.mean(np.diff(lines_x))))
step_y = np.int32(np.round(np.mean(np.diff(lines_y))))

print(lines_x)
print(lines_y)
print(step_x)
print(step_y)

padr_x = 0
padl_x = 0
padr_y = 0
padl_y = 0

if lines_x[0] - step_x < 0:
    padl_x = np.abs(lines_x[0] - step_x)
if lines_x[-1] + step_x > image.shape[1] - 1:
    padr_x = np.abs(lines_x[-1] + step_x - image.shape[1])
    
if lines_y[0] - step_y < 0:
    padl_y = np.abs(lines_y[0] - step_y)
if lines_y[-1] + step_y > image.shape[0] - 1:
    padr_y = np.abs(lines_y[-1] + step_y - image.shape[0])
    
print(padl_x)
print(padr_x)
print(padl_y)
print(padr_y)
#
#

img = np.pad(gray_img1,((padl_y,padr_y),(padl_x,padr_x)),mode='edge')
#print(img)
print(img.shape)
#print(img.shape)

setsx = np.hstack([lines_x[0]-step_x, lines_x, lines_x[-1]+step_x]) + padl_x
setsy = np.hstack([lines_y[0]-step_y, lines_y, lines_y[-1]+step_y]) + padl_y
print(setsx)
print(setsy)

img = img[setsy[0]:setsy[-1],setsx[0]:setsx[-1]]
print(img.shape)

#img = np.copy(gray_img1)
squares = np.zeros([np.round(step_y), np.round(step_x), 64],dtype=np.uint8)
print(squares.shape)
for i in range(0,8):
    for j in range(0,8):
        x1 = setsx[i]
        x2 = setsx[i+1]
        padr_x = 0
        padl_x = 0
        padr_y = 0
        padl_y = 0
            
        if(x2-x1) > step_x:
            if i==7:
                x1 = x2 - step_x
            else:
                x2 = x1 + step_x
        elif(x2-x1) < step_x:
            if i==7:
                padr_x = step_x - (x2-x1)
            else:
                padl_x = step_x - (x2-x1)
        
        y1 = setsy[j]
        y2 = setsy[j+1]
        
        if(y2-y1) > step_y:
            if j==7:
                y1 = y2 - step_y
            else:
                y2 = y1 + step_y
        elif(y2-y1) < step_y:
            if j==7:
                padr_y = step_y - (y2 - y1)
            else:
                padl_y = step_y - (y2 - y1)
                
        print(padl_x)
        print(padr_x)
        print(padl_y)
        print(padr_y)
        squares[:,:,(i+j*8)] = np.pad(img[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)),mode='edge')
        print(x1,x2,y1,y2)
        print(img[y1:y2, x1:x2].shape)
        #squares[:,:,(7-j)*8+i] = img[y1:y2, x1:x2]
        #plt.imshow(squares[:][:][(7-j)*8+i])
        #plt.show()        
                
#cv2.imshow(squares)
for i in range(0,64):
    cv2.imshow('image1',squares[:,:,i])
    cv2.waitKey(0)
    #plt.show()
        

cv2.destroyAllWindows()