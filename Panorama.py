import cv2
import numpy as np
import math
#--------------------------------------------------------------------------------------------------------------------------------------------
def Homography(pt1,pt2,img2):
    H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
    H = np.linalg.inv(H)
    return(H)
#--------------------------------------------------------------------------------------------------------------------------------------------
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img
#--------------------------------------------------------------------------------------------------------------------------------------------
def new_stich(img1,img2):
    indx1=np.where((img1[:,100:-10]!=0))
    ind2 = np.where((img1[:,110:-1]==0))
    img1[ind2]=img2[ind2]
    #img1[-10:-1] = cv2.medianBlur(img1[-10:-1],5)
    return img1
#---------------------------------------------------------------------------------------------------------------------------------------------
def new_stitch_front(img1,img2):
    ind1=np.where((img2[:,0:-20]!=0))
    img1[ind1]=img2[ind1]
    #img1[-20:-1]
    return img1
#----------------------------------------------------------------------------------------------------------------------------------------------

def cylindrical_wrap(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    color = img[y,x]
    theta= (x- K[0][2])/foc_len # angle theta
    h = (y-K[1][2])/foc_len # height
    p = np.array([np.sin(theta),h,np.cos(theta)])
    p = p.T
    p = p.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    return cylinder
#--------------------------------------------------------------------------------------------------------------------------------------------
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
#---------------------------------------------------------------------------------------------------------------------------------------
#Read img1 and store its height
PATH='/home/danish/images/27.png'
img1=cv2.imread(PATH)
#img1 = cv2.medianBlur(img1,5)
height=img1.shape[0]
#Camera Caliberation intrinsic parameters
k = np.array([[951.15755604, 0, 621.51429561], [0, 947.48375342, 349.79574375], [0, 0, 1]])
noi=4000
mean_=[]
print('Size of img1',img1.shape)
img3 = img1.copy()
#initialize Canvas
print(img1.shape)
img1=cylindrical_wrap(img1,k)
canvas=np.zeros((height+200,int(4.3*3.14*(k[0][0]+k[0][1])/2),3),np.uint8)
print('Shape of Canvas is',canvas.shape)
canvas[0:img1.shape[0],0:img1.shape[1]]=img1
#cv2.imshow('canvas',canvas)
H=[]
sum_=0
print('--------------------------------------------------------------------------')
for i in range(28,noi,5):
        img1 = img3.copy()
        #img1 = cv2.GaussianBlur(img1,(5,5),0)
        #img1 = cv2.medianBlur(img1,5)
        img2=cv2.imread('/home/danish/images/'+str(i)+'.png')
        #img2 = cv2.medianBlur(img2,5)
        print(img2.shape)
        #cv2.imshow('img2',img2)
        #cv2.waitKey(0)
        img2=cylindrical_wrap(img2,k)
        print('***************')
        print('Input img',i,'successfully')
        gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
        #Create SIFT object for each image and creating keypoints and descriptors
        sift=cv2.xfeatures2d.SIFT_create()
        kp1,desc1=sift.detectAndCompute(gray1,None)
        kp2,desc2=sift.detectAndCompute(gray2,None)
        bf=cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good=[]
        #Since knn=2 finding best matches for ratio less than 0.75
        for m,n in matches:
            if m.distance<0.75*n.distance:
                good.append(m)
        good=sorted(good,key= lambda x:x.distance)
        good=good[0:30]
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        #print(pt1.shape,pt2.shape)
        p = pt1-pt2
        dist = np.mean(p,axis = 0)
        mean_.append(dist)
        std_y = np.std(p[:,0,1])
        std_x = np.std(p[:,0,0])
        print('Frame No = {} Deviation in X = {} ,Y = {}'.format(i,std_x,std_y))
        if std_y <3 and std_x <3:
            M = np.array([[1,0,dist[0,0]],[0,1,dist[0,1]], [0.,0.,1.]])
            sum_=sum_+dist[0,0]
            print(dist[0,0])
            img3 = cv2.warpPerspective(img2,M,(canvas.shape[1],canvas.shape[0]),flags = cv2.INTER_LINEAR,borderMode = cv2.BORDER_CONSTANT,borderValue = (0,0,0))#flags = cv2.INTER_LINEAR
            #cv2.imshow('img3',img3)
            print(img3.shape)
            print(canvas.shape)
            print('Total displacement',sum_,'\nDisplacement of this frame ',dist[0][0])
            #print(img.shape)
            #cv2.waitKey(0)
            canvas = new_stich(canvas,img3)
            cv2.imshow('canvas',canvas)
            cv2.imwrite('Final_Panorama.jpg',canvas)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
cv2.destroyAllWindows
#print('Mean of all the distances',mean(mean_))
#print('Standar deviation',stddev(mean_))
'''
H=H.append(Homography(pt1,pt2,img2))
img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+gray2.shape[1],height))
img3[0:gray1.shape[0],0:gray1.shape[1]] = img1
        
'''
        


