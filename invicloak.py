import cv2
import time
import numpy as np

print("Select a colour : ")
print("Black(bl)")
print("Red(r)")
print("Blue(b)")
colour = input("=> ")
colour = str.lower(colour)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter("Output.avi",fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0



for i in range(60):
    ret,bg = cap.read() 

bg = np.flip(bg,axis=1)

while (cap.isOpened()):
    ret,img = cap.read()

    if not ret :
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if(colour == "bl"):
        upper_black = np.array([350,55,100])
        lower_black = np.array([0,0,0])
        mask1 = cv2.inRange(hsv,lower_black,upper_black)
        upper_black = np.array([350,55,100])
        lower_black = np.array([0,0,0])
        mask2 = cv2.inRange(hsv,lower_black,upper_black)
    elif(colour == "b"):
        lower_blue = np.array([80,120,50])
        upper_blue = np.array([130,255,255])
        mask1 = cv2.inRange(hsv,lower_blue,upper_blue)
        lower_blue = np.array([80,120,50])
        upper_blue = np.array([130,255,255])
        mask2 = cv2.inRange(hsv,lower_blue,upper_blue)
    elif(colour == "r"):
        lower_red = np.array([0,120,50])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
    mask1 = mask1+mask2
    kernel = np.ones((5,5),np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,kernel)
    mask2 = cv2.bitwise_not(mask1)
    
    res1 = cv2.bitwise_and(img,img,mask=mask2)
    res2 = cv2.bitwise_and(bg,bg,mask=mask1)

    final_o = cv2.addWeighted(res1,1,res2,1,0)
    output.write(final_o)
    cv2.imshow("MAGIC :)",final_o)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()