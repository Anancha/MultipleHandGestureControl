
import cv2
import time
import os
import HandTrackingModule as htm


wCam, hCam = 720, 640

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



while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList) 

    if len(lmList) != 0:

        if lmList[8][2] < lmList[6][2]:
            print("Index Finger open")
            



    h,w,c = overlaylist[0].shape
    img[0:h, 0:w] = overlaylist[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()