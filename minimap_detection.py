import cv2
import numpy as np
import mss
import time

MM_TL_TEMPLATE = cv2.imread("assets/minimap_tl_template.png", 0)  # 좌상단 템플릿
MM_BR_TEMPLATE = cv2.imread("assets/minimap_br_template.png", 0)  # 우하단 템플릿

LOWER_YELLOW = np.array([25, 230, 230])
UPPER_YELLOW = np.array([30, 255, 255])

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

def find_character_process(shared_data):
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
                
                # 예외 처리 추가
                try:
                    if minimap is not None and minimap.size > 0:
                        cv2.imshow("Minimap", minimap)
                    else:
                        print("[DEBUG] 미니맵 이미지가 비어 있습니다.")
                except cv2.error as e:
                    print(f"[ERROR] OpenCV error: {e}")
            
            # 결과 데이터를 큐에 전송
            shared_data["minimap_matching_results"] = detected_char
            cv2.waitKey(1)
            
            # 종료 조건 체크
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()