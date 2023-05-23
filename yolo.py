import cv2
import torch
import pandas as pd
from PIL import Image
import pygame

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\정경오\Desktop\yolo_person_ai\weights\best.pt')

# 외부 웹캠에 연결하기
cap = cv2.VideoCapture(1)

# 경고 사이렌 사운드 파일 재생을 위한 초기화
pygame.mixer.init()
siren_sound = pygame.mixer.Sound('tkdlfps.mp3')  # 경고 사이렌 사운드 파일 경로

# 바운딩 박스 개수 기준 및 경고 사이렌 재생 여부 설정
num_boxes_threshold = 5  # 일정 개수 이상의 바운딩 박스가 탐지되어야 함
play_siren = False

while True:
    # 외부 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 PIL 이미지로 변환
    image = Image.fromarray(frame[:, :, ::-1])

    # 이미지를 YOLOv5 모델로 입력하여 객체 탐지 수행
    results = model(image)

    # 탐지 결과를 pandas 데이터프레임으로 변환
    df = results.pandas().xyxy[0]

    # 탐지된 객체의 바운딩 박스를 웹캠 화면에 그리기
    num_boxes = 0  # 바운딩 박스 개수 초기화

    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 라벨과 신뢰도 텍스트 출력
        label_text = f'{label}: {confidence:.2f}'
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 일정 기준 conf 값 이상인 바운딩 박스 개수 세기
        if confidence >= 0.5:
            num_boxes += 1

    # 바운딩 박스 개수가 기준 개수 이상일 때 경고 사이렌 재생
    if num_boxes >= num_boxes_threshold and not play_siren:
        play_siren = True
        pygame.mixer.music.stop()  # 다른 음악이 재생 중인 경우 중지
        siren_sound.play()

    # 바운딩 박스 개수가 기준 개수 미만일 때 경고 사이렌 중지
    if num_boxes < num_boxes_threshold and play_siren:
        play_siren = False
        pygame.mixer.music.stop()

    # 웹캠 화면 출력
    cv2.imshow('Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 경고 사이렌 관련 리소스 정리
pygame.mixer.quit()

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
