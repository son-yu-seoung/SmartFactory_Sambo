import cv2
import datetime
import time
from Module import crop

while(True):
    try:
        print('▶▶▶삼보산업 PH 예측기입니다.◀◀◀')
        print('[0] : 프로그램 종료, [1]: PH 예측시작(종료시 q버튼), [2] : ~, ')
        choice = int(input('번호를 선택하십시오 : '))
        print()

    except Exception as ex:
        print('error :', ex)
        print()
        choice = None
        time.sleep(1)

    if choice == None:
        continue
    
    elif choice == 0:
        print('프로그램이 3초뒤 종료됩니다.')
        time.sleep(3)
        break

    elif choice == 1:
        path = './Module/realT_img/'
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        hour = 0
        min = 0 
        start = time.time()

        while(True):
            key = cv2.waitKey(1)

            ret, frame = cap.read()
            cv2.imshow('Original Video', frame) # 캡쳐를 할 때마다 불러서 실행시키면 성능이 안좋을 것 같다 -> 나중에 삭제
            
            sec = int(time.time()-start)

            if sec >= 60:
                temp_m= sec // 60
                min += temp_m
                start = time.time() # 이 과정에서 n초의 손실 발생 
                print('run time : {}시간 {}분\n'.format(hour, min))
            
            if min >= 60: 
                hour += 1
                min = 0
                print('run time : {}시간 {}분\n'.format(hour, min))
            
            if min % 5 == 1: # 프로그램을 시작하면 1분째에 한 번 실행한다.(5분주기)
                print('{}분이 경과하였습니다. PH 예측을 시작합니다...\n'.format(5))

                for i in range(7):
                    ret, frame = cap.read()
                    file = path + str(i+1) + '.jpg'
                    cv2.imwrite(file, frame)
                    time.sleep(10) # 초 단위 
                print('image 저장완료'.format(i+1))
                
                crop.readPath(path) # crop
                print('image crop 완료\n')
                # LeNet_predict(path) # PH 예측 

                print('예측 PH = {}\n')

            if key == ord('q') or key == ord('Q'):
                print('프로그램이 3초뒤 종료됩니다.')
                time.sleep(3)

                cap.release()
                cv2.destroyAllWindows()
                break
        print('break')
        break



                
                


