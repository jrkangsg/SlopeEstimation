# Slope Estimation

## Depth Anything 과 Road Segmetation등의 딥러닝 모델을 활용한 Slope Estimation

 - depth_anyting_metric_depth_outdoor.pth, model_epoch3.pth 가중치 파일 필요
  
 - my_test/input에 원하는 영상 저장
 - metric_depth/depth_to_slope_video_Kalman.ipynb 실행

## SIFT 알고리즘을 활용한 Slope Estimation

 - 프레임 단위로 특징점을 추출하고, 이를 통해 현재 경사도 변화를 예측함


## Kalman Filter를 활용한 결과값 보정

 - Kalman 필터를 활용하여 노이즈 제거 및 결과 값의 안정성 개선
