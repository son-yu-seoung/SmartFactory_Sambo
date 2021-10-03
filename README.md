# SmartFactory_Sambo
> __도금수세 작업을 하는 과정에서 깨끗한 수세수가 점점 산성을 띄며 색이 탁해지고 더이상 수세작업을 할 수 없는 폐수가 된다. 이경우 사람이 직접 폐수를 버리고 새로운 수세수를 환수하는 작업을 거쳐야했다. 매번 사람이 환수를 해야하는 시스템은 효율적이지 않기에 웹캠으로 수세수의 이미지를 캡처 후 ph를 예측한 뒤 자동으로 환수되는 시스템을 개발하는 것이 목적이다.__

## LineDetection
웹캠을 수세통위에 설치한 후 절대좌표를 지정하여 crop을 진행하여 PH 예측 모델을 학습시킬 수 있다.
하지만 모든 웹캠이 수세통을 찍는 각도가 일정하지 않고 수세 작업이나 사람에 의한 불안요소가 있기때문에 
수세통을 기준으로 수세수만 crop할 수 있는 LineDetection 모델을 개발하였다.

:+1: :sparkles: :camel: :tada: 
:rocket: :metal: :octocat:
+ **작동 예시**

![LineDetection ](https://user-images.githubusercontent.com/65440674/131129277-061d9fac-ec93-4a32-acef-709a78c600df.gif)

## LeNet
도금 수세 작업을 하는 과정에서 깨끗한 수세수가 점점 산성을 띄며 색이 탁해지는 경우 환수를 진행 해야되는데 그 작업을 자동으로 하기 위해서는 
환수를 진행해야되는 시점을 판단해야되는데 그 판단을 하기위해서 LineDetection 모델을 사용해서 crop한 이미지를 사용해서 PH 예측 모델을 LeNet모델의 구조를 학습시켜서 만들었다.

:+1: :sparkles: :tada: 
:rocket: :metal: :+1:

논문작업 
