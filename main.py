import numpy as np 
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
from time import sleep
from pynput.keyboard import Controller

kb=Controller()


cap=cv2.VideoCapture(0)
# here we set the size of the screen according to the keyboard
cap.set(3,400)
cap.set(4,350)
# here we need the handdetector that detect the hand in the camera
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
detector=HandDetector(detectionCon=0.8)
text=" "
# here we make the function that draw all the keys basically it compile all the things
def drawAll(video,buttonList,hand_landmarks_list,text_):
    for button in buttonList:

        x, y=button.pos
        w, h=button.size
        # self.text=text
        cv2.rectangle(video,button.pos,(x+w,y+h),(0,0,0),cv2.FILLED)
        cv2.putText(video,button.text,(x+9,y+21),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x, y = int(index_finger_tip.x * video.shape[1]), int(index_finger_tip.y * video.shape[0])
            x1, y1 = int(index_finger_mcp.x * video.shape[1]), int(index_finger_mcp.y * video.shape[0])
            x2, y2 = int(thumb_ip.x * video.shape[1]), int(thumb_ip.y * video.shape[0])
            l=np.sqrt(((x2-x1)**2)+((y2-y1)**2))
            for button in buttonList:
                bx, by = button.pos
                bw, bh = button.size
                if bx < x < bx + bw and by < y < by + bh:
                    cv2.rectangle(video, (bx, by), (bx + bw, by + bh), (255, 0, 0), cv2.FILLED)
                    cv2.putText(video, button.text, (bx + 9, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    if l <25:
                        text_+=button.text
                        kb.press(button.text)
                        cv2.rectangle(video, (bx, by), (bx + bw, by + bh), (0, 255, 0), cv2.FILLED)
                        cv2.putText(video, button.text, (bx + 9, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),1)
                        sleep(.2)
                    break
    return text_

# here we make the class for the button so that we make the button class,and define function in it
class button():
    def __init__(self,pos,text,size=(30,30)):
        self.pos=pos
        self.text=text
        self.size=size
            
        # here we define the general function for maiking the soo many buttons
    # def draw(self,video):
        # x, y=self.pos
        # w, h=self.size
        # # self.text=text
        # cv2.rectangle(video,self.pos,(x+w,y+h),(0,0,0),cv2.FILLED)
        # cv2.putText(video,self.text,(x+9,y+21),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        # return
# myButton=button((40,40),"A")
# myButton1=button((80,40),"S")
# myButton2=button((120,40),"D")
# myButton3=button((160,40),"F")
# myButton4=button((40,40),"G")



# here we make the list of the key that is paasing in the loop for making the keys of the keyboard

keys=[["Q","W","E","R","T","Y","U","I","O","P"],
      ["A","S","D","F","G","H","J","K","L",":"],
      ["Z","X","C","V","B","N","M","<",">","/"]]


# we makr the list of button and append the button in it

buttonList=[]
for j in range(0,3):
    for i,key in enumerate(keys[j]):
        buttonList.append(button((40*i+100,40+j*40),key))      
       

while cap.isOpened():
    success,video=cap.read()
    video=cv2.flip(video,1)

    # detecting the two things
    results = hands.process(video)
    # Extract hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on image
            mp_drawing.draw_landmarks(video, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # text=draw_all(image, button_list, results.multi_hand_landmarks,text)
    # here we make the button of the rectangle by using the cv2 rectangle
    # cv2.rectangle(video,(40,40),(70,70),(0,0,0),cv2.FILLED)
    # cv2.putText(video,"A",(49,61),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

# now we initializing the button with the help of for loop
# shiftting the below code outside the loop so where we compile all the things into ones
    # for j in range(0,3):
    #     for i,key in enumerate(keys[j]):
    #         buttonList.append(button((40*i+100,40+j*40),key))
    # VideoCapture=myButton.draw(video)
    # VideoCapture=myButton1.draw(video)
    # VideoCapture=myButton2.draw(video)
    # VideoCapture=myButton3.draw(video)

    newtext=drawAll(video, buttonList,results.multi_hand_landmarks,text)
    cv2.rectangle(video, (55,250), (200,300), (255, 0, 0), cv2.FILLED)
    cv2.putText(video, newtext, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    print(newtext)
    
   



        #         cv2.rectangle(video,button.pos,(x+w,y+h),(0,0,0),cv2.FILLED)
        # cv2.putText(video,button.text,(x+9,y+21),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

             

    cv2.imshow("VitualKeyboard",video)
    # cv2.imshow("camera",img)
    
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
        