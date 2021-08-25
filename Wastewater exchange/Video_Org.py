import cv2
import datetime
import time

path = './Module/image/'

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 종료할 때 [WARN:0]오류 제거 
# print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
n = 1 

while(True):
    
    ret, frame = cap.read() # ret은 정상적으로 작동하는지 반환(True, False), frame은 이미지를 넘파이배열 형대로 가져옴 
    # 비디오 frame의 크기를 설정 (너비, 높이):절대크기, fx=0.3, fy=0.7:상대크기, INTER_(LINEAR, AREA, ...etc)
    # frame =cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR) # 비디오 영상의 크기 설정
    #frame = cv2.resize(frame, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Original Video', frame)

    key = cv2.waitKey(1) # 키를 기다리는 대기 함수 0이면 무한대기 1000이면 1초

    if key == ord('q') or key == ord('Q'):
        break

    # elif key == ord('s') or key == ord('S'):
    #     file = path + 'green_' + '7.59_' + str(n) + '.jpg'
    #     n += 1 
    #     # file = path + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
    #     cv2.imwrite(file, frame)
    #     print(file, ' saved')
    
    elif key == ord('g') or key == ord('G'):

        for i in range(5): # 10장의 사진을 찍는다.
            ret, frame = cap.read() # frame이 실질적으로 영상을 가져오기 때문에 frame을 reset시켜줘야 움직이는 사진이 찍힌다!
            file = path + 'green_' + '7.24_' + str(n) + '.jpg'
            n +=1
            # file = path + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.jpg'
            cv2.imwrite(file, frame)
            print(file, ' saved')
            time.sleep(0.3) # 0.3초 간격으로

    
cap.release()
cv2.destroyAllWindows()

