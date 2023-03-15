# Image-Morphing
Image Morphing using Python and OpenCV with automatic Feature Detection

Download shape_predictor_68_face_landmarks file from the below link and paste it in same directory 
https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat

***The following commands are only valid for ubuntu***

To run the program,
conda activate <env>
python3 2021CSB1072_Arnav_PA2.py

If you wish to change the images, change variables 'filename1' and 'filename2'

I have used dlib to geneate tie points and save it into .node file
Then the executable 'triangle' takes it and generates coordinates of triangle
That is then processed by Delaunay's triangulation algorithm to generate 101 image files (alpha varying from 0 to 1 with steps of 0.01) saved in images folder
Then using PIL, the image files are converted to GIF format
That is then save as 'output.gif'
