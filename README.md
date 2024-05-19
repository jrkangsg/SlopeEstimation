# Slope Estimation

## Depth Anything 과 Road Segmetation등의 딥러닝 모델을 활용한 Slope Estimation
 ### 실행 방법
 - depth_anyting_metric_depth_outdoor.pth, model_epoch3.pth 가중치 파일 필요
 - my_test/input에 원하는 영상 저장
 - metric_depth/depth_to_slope_video_Kalman.ipynb 실행

 ### 모델 설명
 - 기하학적 방식으로 전방의 경사로의 경사도를 구할 수 있음.
   
 - 이를 위해서는 거리 데이터가 필요.
 - 단안 카메라를 이용한 심도 데이터를 통해 전방 경사도를 예측할 수 있음.
 - 심도데이터의 획득 어려움으로 인해, Large-Scale 데이터를 활용한 Depth Anything 모델을 사용하여, 심도 데이터를 얻음.

 - Road Segmentation을 통해 도로의 심도데이터를 뽑아내고, 보정하여 정확도를 향상할 수 있음.
 - LFD 모델을 사용하여, 연산 속도를 줄여, 실시간성을 보장하도록 함.

## SIFT 알고리즘을 활용한 Slope Estimation

 - 프레임 단위로 특징점을 추출하고, 이를 통해 현재 경사도 변화를 예측함


## Kalman Filter를 활용한 결과값 보정

 - Kalman 필터를 활용하여 노이즈 제거 및 결과 값의 안정성 개선
