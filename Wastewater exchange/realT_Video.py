import cv2, os
import datetime
import time
import predict
from Module import crop

while(True):
    try:
        print('\n▶▶▶▶▶▶삼보산업 PH 예측기입니다.◀◀◀◀◀◀\n')
        print('▶ 1 : PH 예측시작(종료시 q버튼)')
        print('▶ 2 : 기타 ~')
        print('▶ 0 : 프로그램 종료\n')
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
        img_path = './realT_img/'
        pred_path = './realT_crop/'

        pred_model = predict.PH_predict(pred_path)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        os.system('cls')

        print('\n\n▶▶▶▶▶▶PH 측정 프로그램 구동시작◀◀◀◀◀◀\n')
        print('▶▶ 설명')
        print('▶ 1. 프로그램은 구동시작 후 [1분] 뒤 첫 PH 예측을 실행합니다.')
        print('▶ 2. 프로그램은 첫 PH 예측 후 [5분]마다 PH 예측을 실행합니다.')
        print('▶ 3. 프로그램을 종료하시려면 웹캠 화면을 띄운상대로 [\'q\' or \'Q\'] 버튼을 입력하세요.\n')
        
        print('※ PH 예측중에 웹캠의 작동이 중지되는 것은 정상입니다.')
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
                print('[ run time : {}시간 {}분 ]\n'.format(hour, min))
            
            if min >= 60: 
                hour += 1
                min = 0
                print('[ run time : {}시간 {}분 ]\n'.format(hour, min))
            
            if min % 5 == 0: # 프로그램을 시작하면 1분째에 한 번 실행한다.(5분주기)
                print('[ {}분이 경과하였습니다. PH 예측을 시작합니다... ]\n'.format(min))

                for i in range(7):
                    ret, frame = cap.read()
                    file = img_path + str(i+1) + '.jpg'
                    cv2.imwrite(file, frame)
                    time.sleep(1) # 초 단위 
                
                crop.readPath(img_path) # crop
                print('---> 이미지 저장, 자르기 완료'.format(i+1))
                
                PH = pred_model.predict(256) # PH 예측, dim 인자 줘야함 

                print('\n※ 예측 PH = {}\n'.format(PH[:, 0]))

            if key == ord('q') or key == ord('Q'):
                print('프로그램이 3초뒤 종료됩니다.')
                time.sleep(3)

                cap.release()
                cv2.destroyAllWindows()
                break
        print('break')
        break



                
                


