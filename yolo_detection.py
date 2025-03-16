import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time
import os
import glob
import argparse

def yolo_detection_process(shared_data):
    """
    YOLOv8 모델을 이용해 실시간 감지를 수행하는 프로세스.
    모델은 'runs/detect/train16/weights/best.pt'에 저장된 사용자 정의 모델을 사용합니다.
    """
    model_path = "runs/detect/train16/weights/best.pt"
    model = YOLO(model_path)  # YOLOv8 모델 로드
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}  # 기본 모니터 (필요 시 영역 조정)
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            # mss는 BGRA 형식으로 반환하므로 BGR로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # YOLOv8 추론 (conf 임계값 설정)
            results = model(frame, conf=0.8)
            # 결과를 시각화한 이미지 생성 (첫 번째 결과 사용)
            annotated_frame = results[0].plot()
            # YOLO 결과 데이터 (감지된 객체가 없으면 빈 리스트)
            # (results[0].boxes.data가 tensor일 경우 tolist()로 변환)
            boxes = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []
            shared_data["yolo_detection_results"] = boxes
            cv2.waitKey(1)
            
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()