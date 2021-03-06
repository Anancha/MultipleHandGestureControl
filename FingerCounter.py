
import cv2
import time
import os
import HandTrackingModule as htm  # Hand Tracking Module 


wCam, hCam = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlaylist  = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlaylist.append(image)

print(len(overlaylist))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList) 

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
 
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # if the tip is below the middle
                fingers.append(1)
            else:    
                fingers.append(0)
 
        # print(fingers)
        totalFingers = fingers.count(1)               # Count the number of 1's in the list 
        print(totalFingers)                           # Print the number of fingers



        h,w,c = overlaylist[totalFingers-1].shape
        img[0:h, 0:w] = overlaylist[totalFingers-1]

        cv2.rectangle(img, (20, 325), (200, 525), (0, 255, 0), cv2.FILLED)  # Draw a green rectangle to show the number area 
        cv2.putText(img, str(totalFingers), (45, 445), cv2.FONT_HERSHEY_PLAIN,   # text position 
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (600, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  
    # show FPS as text by Color and  position of text (600,70)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()