import cv2 
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    success, img = cap.read()
    hands,img = detector.findHands(img) # with drowing
    #hands = detector.findHands(img, draw=False) # No drawing

    # print(len(hands))   # if you want to show number of hands

    if hands:
        #hand 1 
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        handType1 = hand1["type"]

        #print(len(lmList1),lmList1)
        #print(bbox1)    # if you want to show bbox
        #print(centerPoint1) # center point
        #print(handType1) # hand type

        fingers1  = detector.fingersUp(hand1)
        #length, info, img = detector.findDistance(lmList1[8],lmList1[12],img)



        if len(hands)==2:
            #hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2["center"]
            handType2 = hand2["type"]
            #print(len(lmList2),lmList2)
            #print(bbox2)    # if you want to show bbox
            #print(centerPoint1,centerPoint2) # center point
            #print(handType1,handType2) # hand type

            fingers2  = detector.fingersUp(hand2)
            #print(fingers1,fingers2) # if you want to show fingers 

            #length, info, img = detector.findDistance(lmList1[8],lmList2[8],img) # distance between two hands with drawing
            length, info, img = detector.findDistance(centerPoint1,centerPoint2,img) # center point 


    cv2.imshow("img", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


