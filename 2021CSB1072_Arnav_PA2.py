#!/usr/bin/env python

import numpy as np
import cv2
import sys
import glob
from PIL import Image
import re
import dlib
import subprocess

def readPoints(path) :
    '''This function reads points from a text file'''
    # Create an array of points.
    points = []
    # Read points
    n = 0
    with open(path) as file :
        for line in file :
            if (n == 0):
                n += 1
                continue
            try:
                w, x, y, z = line.split()
                points.append((int(x), int(y)))
            except:
                continue

    return points

def applyAffineTransform(src, srcTri, dstTri, size):
    '''Apply affine transform calculated using srcTri and dstTri to src and output an image of size.'''

    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    '''Warps and alpha blends triangular regions from img1 and img2 to img'''

    # Find bounding rectangle for each triangle
    rect1 = cv2.boundingRect(np.float32([t1]))
    rect2 = cv2.boundingRect(np.float32([t2]))
    rect = cv2.boundingRect(np.float32([t]))

    # Offset points
    t1Rect = []
    t2Rect = []
    tRect = []

    # Get offset points
    for i in range(0, 3):
        tRect.append(((t[i][0] - rect[0]),(t[i][1] - rect[1])))
        t1Rect.append(((t1[i][0] - rect1[0]),(t1[i][1] - rect1[1])))
        t2Rect.append(((t2[i][0] - rect2[0]),(t2[i][1] - rect2[1])))


    # Get mask by filling triangle
    mask = np.zeros((rect[3], rect[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2Rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    
    
    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * ( 1 - mask ) + imgRect * mask

    
def numericalSort(value):
    '''This Function sorts filenames of images for GIF Creation'''
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def make_gif(frame_folder):
    '''This function creates GIF file with input given a folder of images'''
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.jpg"), key=numericalSort)]
    frame_one = frames[0]
    frame_one.save("output.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

#main function
if __name__ == '__main__' :

    part_option = int(input("Option:\n1. Enter 1 to Execute part A\n2. Enter 2 to execute part B\nOption: "))

    print("Starting Program! May take upto 1 min...")

    #Filenames of images
    filename1 = 'donald_trump.jpg'
    filename2 = 'ted_cruz.jpg'

    #Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    landmarks_arr1 = []
    landmarks_arr2 = []


    if(part_option == 2):
        #Get tri points using dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(img1)

        landmarks_arr1= [(0,0), 
                        (0,img1.shape[0]-1), 
                        (img1.shape[1]-1, 0),
                        (img1.shape[1]-1, img1.shape[0]-1)
                        ]

        # traverse through each possible face in faces we got from detector
        for face in faces:
            # landmarks are nothing but the points which is a possible tie points in the image
            landmarks = predictor(img1, face)
            
            # since the shape_predictor uses a dat file of 68 points, we traverse through all those 68 points and add them to an array for delaunay triangulation
            for i in range(68):
                x =  landmarks.part(i).x
                y =  landmarks.part(i).y
                landmarks_arr1.append((x,y ))
                cv2.circle(img1, (x,y), 3,(0,0,255), -1)

        for i in landmarks_arr1:
            img1 = cv2.circle(img1, i, radius=10, color=(0, 0, 255), thickness=-1)

        

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(img2)

        landmarks_arr2= [(0,0), 
                        (0,img2.shape[0]-1), 
                        (img2.shape[1]-1, 0),
                        (img2.shape[1]-1, img2.shape[0]-1)
                        ]

        # traverse through each possible face in faces we got from detector
        for face in faces:
            # landmarks are nothing but the points which is a possible tie points in the image
            landmarks = predictor(img2, face)
            
            # since the shape_predictor uses a dat file of 68 points, we traverse through all those 68 points and add them to an array for delaunay triangulation
            for i in range(68):
                x =  landmarks.part(i).x
                y =  landmarks.part(i).y
                landmarks_arr2.append((x,y ))
                cv2.circle(img2, (x,y), 3,(0,0,255), -1)

        for i in landmarks_arr2:
            img2 = cv2.circle(img2, i, radius=10, color=(0, 0, 255), thickness=-1)

    elif(part_option == 1):
        file_t = open('tie.txt', 'r')

        n = 0
        for line in file_t.readlines():
            if(n == 0):
                n += 1
                continue
            a, b, c, d = line.split()
            landmarks_arr1.append((a, b))
            landmarks_arr2.append((c, d))

        file_t.close()
    #Save the output points as .node file for triangle program to read
    file1 = open(filename1+'.node', 'w')
    file1.write(str(len(landmarks_arr1))+" 2 0 0\n")
    n = 1
    for i in landmarks_arr1:
        a, b = i[0], i[1]
        file1.write(str(n)+' '+str(a)+' '+str(b)+'\n')
        n += 1
    file1.close()

    #for i in range(0,len(landmarks_arr2)):
    #    print(landmarks_arr1[i][0], landmarks_arr1[i][1], landmarks_arr2[i][0], landmarks_arr2[i][1])

    file2 = open(filename2+'.node', 'w')
    file2.write(str(len(landmarks_arr2))+" 2 0 0\n")
    n = 1
    for i in landmarks_arr2:
        a, b = i[0], i[1]
        file2.write(str(n)+' '+str(a)+' '+str(b)+'\n')
        n += 1
    file2.close()

    #Get tri points of a single file only using the triangle program
    subprocess.call("./triangle -Q "+filename1+".node", shell=True)
    subprocess.call("./triangle -Q "+filename2+".node", shell=True)

    #Generate 101 images in folder ./images to make GIF
    for j in range(0,101):
        
        # Alpha
        alpha = 0.01*j

        # Read images
        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)
        
        # Convert Matrix to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points1 = readPoints(filename1 + '.1.node')
        points2 = readPoints(filename2 + '.1.node')
        points = []

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1-alpha)*points1[i][0] + alpha*points2[i][0]
            y = (1-alpha)*points1[i][1] + alpha*points2[i][1]
            points.append((x,y))

        # Final output Matrix
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        # Read triangles from tri.txt
        n = 0
        with open(filename1+'.1.ele') as file :
            for line in file :
                # Skip the first line
                if(n == 0):
                    n += 1
                    continue

                # Try to read line from file
                try:
                    w, x,y,z = line.split()
                except:
                    continue
                
                # Tri points
                x = int(x) - 1
                y = int(y) - 1 
                z = int(z) - 1
                
                # Get triangles
                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [ points[x], points[y], points[z] ]

                # Morph one triangle at a time.
                morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        # Write image to image directory
        cv2.imwrite('./images/h'+str(j+1)+'.jpg',np.uint8(imgMorph))
    
    # Make GIF
    make_gif('./images')

