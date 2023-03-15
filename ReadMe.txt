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
