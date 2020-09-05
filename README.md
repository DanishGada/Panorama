# IMAGE STITCHING IN PLANAR AND CYLINDRICAL COORDINATE SYSTEM

## OVERVIEW
The concept of Image Stitching has been around for quite some time and has been one of the most successful implementations of Computer Vision. These days, panoramas can be shot on every smartphone. In this project, we study and implement some robust and well-establised "Feature Matching" techniques to build a real-time Image Stichting Algorithm. We shall discuss each of the algorithms in brief. The coding here is done in Python-3, using the Numpy framework for optimizing 

## REQUIREMENTS
numpy==1.16.4 opencv-python==3.4.2

## Image Stitching using Homography

1. **Feature Detection on images using SIFT**.
SIFT stands for Scale Invariant Feature Transform, which is used for detection features on the images. The example image taken for representing the action of SIFT is shown as 1.jpeg in the repository.[1] 

![image_screenshot_01 03 2020](https://user-images.githubusercontent.com/45517467/75624246-a6ef8500-5bd8-11ea-9dfc-73cacfb0fd20.png)

2. **Feature Matching between two images using Brute Force KNN Matching**.
Here, we take a descriptor of a feature from the first set and match it with the descriptors of all other features from the second set. Distance is calculated using the L2 norm, and the closest one is returned. We chose to remove the cross-checking in our Algorithm to improve its runtime. Also, instead of drawing all the best matches, we draw k = 2 best matches, that is, drawing two match-lines for each keypoint to improve run time while also making sure that the feature is a Reliable one. Its output with 1.jpeg and 2.jpeg is shown below.[2]

![image](https://user-images.githubusercontent.com/45517467/75624402-caff9600-5bd9-11ea-8c93-6ab62c30a6de.png)

3. **Generation of the Transformation Matrix**.
Planar homography relates the transformation between two planes. It is a 3x3 matrix with 8 DoF. The elements of the matrix are generally normalized to 1. Here, To transfer an image from a source plane to a target plane, we determine a homography matrix and apply it on the source plane, to transfer the image into the target plane, as per the matched features. In this way, we can obtain the result as shown.[3]
![image](https://user-images.githubusercontent.com/45517467/75624566-680efe80-5bdb-11ea-8542-73767db92bf5.png)

In the code, we did the planar image stitching for multiple images. So, we initially made a Black canvas of 5 times the length of the image, and height 200 pixels greater than the height of the image to account for any errors in the vertical direction. After the first two images have been stitched as shown, Using the third image, we calculate the homography matrix between the third image and the second image that had been warped in the first iteration, before it was stitched to the source image. Similarly, the homography matrix between the fourth image and the third image before wraping is calculated. Stitching of all the four images results as shown, 
![image](https://user-images.githubusercontent.com/45517467/75624642-3c404880-5bdc-11ea-9ec7-1600d63b0a35.png) 

## Cylindrical Image Stitching with Translational Model

1. **Conversion from Planar to Cylindrical Coordinates**.
Image Stitching in cylindrical coordinates requires the use of a tripod for recording the video, since then the motion along the y axis will be as good as null. In such a scenerio, we use the translational model for image stitching. But before that, we should convert the image into cylindrical coordinates. According to [4], image can be transferred to a cylinder using the formula listed in the slides. When implemented, its results are as shown,<br>
![image](https://user-images.githubusercontent.com/45517467/75624917-06509380-5bdf-11ea-85d8-4a74d92573fa.png)<br>
Note that conversion from planar to cylindrical coordinates requires the use of the focal length, which can be found using the intrinsic camera parameters. These can be found using camera calibration. To calculate these parameters, we used a chessboard for getting the corner locations of an image. The repository consisting of the source code can be referred to at [5]
> For the image shown above, the intrinsic camera parameters are found, and used for demonstration of the algorithm

2. **Feature Detection on images using SIFT**
Covered in Section 3
3. **Feature Matching between two images using Brute Force KNN**.
Covered in Section 3
4. **Translational Model**.
Now that the feature points are obtained for the pair of images using SIFT, it is time for the translational model for stitching.Here, we have approximated the least mean square technique as used in [6] by taking the difference of the obtained feature points between two consecutive images and take the mean of the result We also find its standard deviation and divide the answer into the standard deviation along x axis and y axis. We then establish a threshold. Only if the value of the standard deviation along both the axes is less than threshold,the perspective warping of the second image with-respect-to the first image shall be performed. This is done for gap closure. After the distance between the Keypoints are calculated as stated, we divide the array into the distance along the x axis and along y axis, which is then fed into the translational matrix. This matrix is used for the perspective warping of the images. The array is similarly updated for each iteration. The translational matrix is generated between the next image and the warped version of the previous image. In this manner, we can obtain the cylindrical panorama.<br>
> The length of the canvas is equal to the circumference of the cylinder, that is, 2* pi * f, where f is the focal length of the camera, extracted from the intrinsic camera parameters, and the height being 200 pixels more than the actual height of the image. 


## REAL - TIME IMPLEMENTATION
The real - time implementation was performed by recording the video of our lab, where the camera was placed roughly at the centre of the room and rotated at a roughly uniformly angular velocity. The camera used was Intel RealSense Depth Module, using pyrealsense2 for recording the video and for camera calibration. The result for this can be seen below

![image](https://i.imgur.com/v3H3wp6.jpg)

## TEAM MEMBERS

1. Saurabh Kemekar
2. Arihant Gaur
3. Pranav Patil
4. Danish Gada

## PROJECT MENTOR
1. Aman Jain

## REFERENCES

[1] Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94 <br />
[2] Jakubovic, Amila & Velagic, Jasmin. (2018). Image Feature Matching and Object Detection Using Brute-Force Matchers. 83-86. 10.23919/ELMAR.2018.8534641. </br>
[3] http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf <br />
[4] http://cs.brown.edu/courses/cs129/results/final/yunmiao/ <br />
[5] https://github.com/saurabhkemekar/camera-calibration <br />

