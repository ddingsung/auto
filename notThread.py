import cv2
import numpy as np
import mss
import os
import glob
from ultralytics import YOLO
import multiprocessing
import time
import argparse

# 미니맵 관련 템플릿 이미지 로드 (경로에 맞게 조정)
MM_TL_TEMPLATE = cv2.imread("assets/minimap_tl_template.png", 0)  # 좌상단 템플릿
MM_BR_TEMPLATE = cv2.imread("assets/minimap_br_template.png", 0)  # 우하단 템플릿

# 캐릭터 감지를 위한 HSV 필터 (노란색)
LOWER_YELLOW = np.array([25, 230, 230])
UPPER_YELLOW = np.array([30, 255, 255])

# 경로 설정 관련 변수
floor_paths = {}      # {floor_number: (start_point, end_point)}
temp_points = []      # 현재 층의 임시 클릭 좌표
current_floor = 1     # 현재 층 번호

def template_matching_process(queue):
    """
    templates 폴더 내의 모든 PNG 템플릿 이미지를 불러와
    캡처한 화면에서 템플릿 매칭을 수행하는 프로세스.
    """
    template_dir = "templates"
    template_paths = glob.glob(os.path.join(template_dir, "*.png"))
    templates = []
    if len(template_paths) == 0:
        print(f"No template images found in {template_dir}")
        return
    for path in template_paths:
        template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template_img is not None:
            templates.append((os.path.basename(path), template_img))
        else:
            print(f"Failed to load template image: {path}")
    print(f"Loaded {len(templates)} template(s).")
    
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}  # 기본 모니터 (필요 시 영역 조정)
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            
            detected_centers = []  # 매칭된 사각형 중심 좌표 리스트
            
            # 각 템플릿에 대해 매칭 수행
            for tpl_name, tpl_img in templates:
                w, h = tpl_img.shape[::-1]
                res = cv2.matchTemplate(gray_frame, tpl_img, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  # 임계값 (필요에 따라 조절)
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):  # (x, y) 좌표
                    # 매칭 영역 그리기
                    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                    cv2.putText(frame, tpl_name, (pt[0], pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # 중심 좌표 계산 (정수형)
                    center_x = pt[0] + w // 2
                    center_y = pt[1] + h // 2
                    detected_centers.append((center_x, center_y))
            
            cv2.imshow("Template Matching", frame)
            # 결과 데이터에 중심 좌표 리스트를 담아서 보냄
            results_data = {"source": "template_matching", "bbox": detected_centers}
            queue.put(results_data)
            time.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def find_minimap(screen_gray):
    tl_result = cv2.matchTemplate(screen_gray, MM_TL_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    br_result = cv2.matchTemplate(screen_gray, MM_BR_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, max_tl_val, _, max_tl_loc = cv2.minMaxLoc(tl_result)
    _, max_br_val, _, max_br_loc = cv2.minMaxLoc(br_result)
    print(f"[DEBUG] 미니맵 템플릿 매칭 결과: tl_val={max_tl_val:.2f}, br_val={max_br_val:.2f}")
    if max_tl_val >= 0.7 and max_br_val >= 0.7:
        mm_x, mm_y = max_tl_loc
        mm_w = (max_br_loc[0] - mm_x) + 10
        mm_h = max_br_loc[1] - mm_y
        print(f"[DEBUG] 미니맵 위치: {(mm_x, mm_y, mm_w, mm_h)}")
        return mm_x, mm_y, mm_w, mm_h
    return None

def find_character_in_minimap(minimap):
    if minimap is None or minimap.size == 0:
        print("[DEBUG] 미니맵 이미지가 비어 있습니다.")
        return None
    hsv_minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_minimap, LOWER_YELLOW, UPPER_YELLOW)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"[DEBUG] 미니맵 캐릭터 감지: x={x}, y={y}, w={w}, h={h}")
        return x, y, w, h  
    print("[DEBUG] 미니맵에서 캐릭터를 찾지 못했습니다.")
    return None

def find_character_process(queue):
    """
    미니맵에서 캐릭터를 감지하는 프로세스.
    결과를 queue를 통해 전달합니다.
    """
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}
        
        while True:
            # 화면 캡처
            frame = np.array(sct.grab(monitor))
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 미니맵 찾기
            screen_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            minimap_pos = find_minimap(screen_gray)

            detected_char = None
            if minimap_pos:
                mm_x, mm_y, mm_w, mm_h = minimap_pos
                
                # 미니맵 부분만 추출
                minimap = frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w]
                
                # 미니맵에서 캐릭터 찾기
                char_pos = find_character_in_minimap(minimap)
                
                if char_pos:
                    cx_m, cy_m, cw_m, ch_m = char_pos
                    abs_cx = mm_x + cx_m + (cw_m // 2)
                    abs_cy = mm_y + cy_m + (ch_m // 2)
                    detected_char = (abs_cx, abs_cy)

                    # 캐릭터 영역 표시
                    cv2.rectangle(minimap, 
                                  (cx_m, cy_m), 
                                  (cx_m + cw_m, cy_m + ch_m), 
                                  (255, 0, 255), 2)
                
                # 미니맵만 표시
                cv2.imshow("Minimap", minimap)

            # 결과 데이터를 큐에 전송
            results_data = {
                "source": "minimap_matching",
                "bbox": [detected_char] if detected_char else []
            }
            queue.put(results_data)
            
            # 종료 조건 체크
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    
    cv2.destroyAllWindows()

def yolo_detection_process(queue):
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
            results_data = {"source": "yolo_detection", "bboxes": boxes}
            queue.put(results_data)
            time.sleep(0.01)
            
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def aggregator(queue):
    """
    공유 큐에 들어오는 감지 결과들을 읽어서 출력합니다.
    """
    while True:
        if not queue.empty():
            data = queue.get()
            if data["source"] == "template_matching":
                bboxes = data.get("bbox", [])
                if bboxes:
                    for i, box in enumerate(bboxes):
                        print(f"Template Matching - 객체 {i+1} 중심 좌표: {box}")
                else:
                    print("Template Matching - 감지된 객체 없음")
            elif data["source"] == "yolo_detection":
                bboxes = data.get("bboxes", [])
                if bboxes:
                    for i, box in enumerate(bboxes):
                        print(f"YOLO Detection - 객체 {i+1} 좌표: {box}")
                else:
                    print("YOLO Detection - 감지된 객체 없음")
            elif data["source"] == "minimap_matching":
                bboxes = data.get("bbox", [])
                if bboxes:
                    for i, box in enumerate(bboxes):
                        print(f"Minimap Detection - 캐릭터 중심 좌표: {box}")
                else:
                    print("Minimap Detection - 캐릭터를 찾을 수 없음")
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="프로세스 선택")
    parser.add_argument("--mode", type=int, choices=[1,2,3], required=True,
                        help="1: 템플릿 매칭, 2: YOLO 감지, 3: 미니맵 감지")
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn')
    shared_queue = multiprocessing.Queue()
    
    # 선택한 모드에 따라 해당 프로세스와 aggregator를 실행합니다.
    if args.mode == 1:
        p_template = multiprocessing.Process(target=template_matching_process, args=(shared_queue,))
        p_aggregator = multiprocessing.Process(target=aggregator, args=(shared_queue,))
        p_template.start()
        p_aggregator.start()
        p_template.join()
        p_aggregator.join()
    elif args.mode == 2:
        p_yolo = multiprocessing.Process(target=yolo_detection_process, args=(shared_queue,))
        p_aggregator = multiprocessing.Process(target=aggregator, args=(shared_queue,))
        p_yolo.start()
        p_aggregator.start()
        p_yolo.join()
        p_aggregator.join()
    elif args.mode == 3:
        p_minimap = multiprocessing.Process(target=find_character_process, args=(shared_queue,))
        p_aggregator = multiprocessing.Process(target=aggregator, args=(shared_queue,))
        p_minimap.start()
        p_aggregator.start()
        p_minimap.join()
        p_aggregator.join()