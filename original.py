# 층 인식 전 백업 코드
import cv2
import numpy as np
import mss
import time
import ctypes
import keyboard
import pyautogui  # pyautogui도 같이 사용
import threading  # 스레딩 추가
from ctypes import wintypes
from ultralytics import YOLO
import random  # Add import for random
import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel
from PIL import Image, ImageTk
import os

#########################################
# 전역 변수 초기화
#########################################
character_template = None
character_template_selected = False
current_direction = None
threshold = 5  # t3.py에서 가져온 값 추가
press_tolerance = 10
held_key = None
shift_pressed = False  # Shift 키 상태 변수 추가
monster_direction = None  # 몬스터 감지 방향 변수 추가
key_thread_running = True  # 키 입력 스레드 제어 변수
last_key_update = 0  # 마지막 키 업데이트 시간
key_delay = 0.1  # 키 업데이트 최소 간격 (초)
prioritize_minimap = True  # 미니맵 이동 우선 (True) 또는 몬스터 감지 우선 (False)
debounce_counter = 0  # 디바운싱 카운터 추가
last_held_key = None  # 마지막 키 상태 저장
key_debounce_timer = None  # 디바운싱 타이머
monster_reset_timer = None  # 몬스터 리셋 타이머
last_monster_time = 0  # 마지막 몬스터 감지 시간
monster_verify_timer = None  # 몬스터 재확인 타이머
monster_detected_in_frame = False  # 현재 프레임에서 몬스터 감지 여부

# DirectInput 상수
DIK_LEFT = 0xcb
DIK_RIGHT = 0xcd
DIK_SHIFT = 0x2a

# SendInput 함수 정의
SendInput = ctypes.windll.user32.SendInput

# InputType
INPUT_KEYBOARD = 1

# KEYBDINPUT 구조체
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

# INPUT 구조체
class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki", KEYBDINPUT),
        ("padding", ctypes.c_ubyte * 8)
    ]

# 키 입력 플래그
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

# DirectInput을 사용한 키 이벤트 함수들
def press_key_di(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(0, key_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
    )
    SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))

def release_key_di(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(0, key_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
    )
    SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))

# 키 입력 스레드 함수
def key_pressing_thread():
    global key_thread_running, held_key, shift_pressed
    last_dir_key = None
    last_shift_state = False
    refresh_counter = 0
    
    while key_thread_running:
        try:
            current_dir_key = held_key  # 현재 방향키 상태
            current_shift = shift_pressed  # 현재 쉬프트 상태
            
            # 키 상태가 변경되었는지 확인
            if current_dir_key != last_dir_key:
                # 이전 키 해제
                if last_dir_key:
                    keyboard.release(last_dir_key)
                    release_key_di(DIK_LEFT if last_dir_key == 'left' else DIK_RIGHT)
                    print(f"[키 스레드] {last_dir_key} 키 해제")
                
                # 새 키 누름
                if current_dir_key:
                    keyboard.press(current_dir_key)
                    press_key_di(DIK_LEFT if current_dir_key == 'left' else DIK_RIGHT)
                    print(f"[키 스레드] {current_dir_key} 키 누름")
                
                last_dir_key = current_dir_key
            
            # 쉬프트 키 상태 확인
            if current_shift != last_shift_state:
                if current_shift:
                    keyboard.press('shift')
                    press_key_di(DIK_SHIFT)
                    print("[키 스레드] Shift 키 누름")
                else:
                    keyboard.release('shift')
                    release_key_di(DIK_SHIFT)
                    print("[키 스레드] Shift 키 해제")
                
                last_shift_state = current_shift
            
            # 주기적으로 키 상태 리프레시 (0.5초마다) - 키 누름 상태가 풀리는 것 방지
            refresh_counter += 1
            if refresh_counter >= 10:  # 10번의 0.05초 = 0.5초
                refresh_counter = 0
                
                # 방향키 리프레시
                if current_dir_key:
                    keyboard.press(current_dir_key)
                    press_key_di(DIK_LEFT if current_dir_key == 'left' else DIK_RIGHT)
                
                # 쉬프트 키 리프레시
                if current_shift:
                    keyboard.press('shift')
                    press_key_di(DIK_SHIFT)
            
            time.sleep(0.05)  # 20Hz로 실행
        except Exception as e:
            print(f"[키 스레드] 오류 발생: {e}")
            time.sleep(0.1)

# 키 입력 업데이트 함수 (디바운싱 포함)
def update_key(key, source="unknown"):
    global held_key, last_key_update, debounce_counter, last_held_key, key_debounce_timer
    
    current_time = time.time()
    
    # 프로그램 종료 시 즉시 키 해제
    if source == "program_exit":
        if held_key:
            keyboard.release(held_key)
            print(f"[키 입력] 프로그램 종료로 {held_key} 키 해제")
        held_key = None
        return
    
    # 디바운싱: 키 변경이 너무 빠르게 일어나지 않도록 함
    if (current_time - last_key_update) < key_delay:
        debounce_counter += 1
        if debounce_counter % 10 == 0:  # 로그 과다 출력 방지
            print(f"[키 입력] 디바운싱: {key} ({source}), 카운터: {debounce_counter}")
        return
    
    # 몬스터가 감지되었고 미니맵 소스에서 키 업데이트가 왔을 때 무시
    if monster_direction is not None and source == "minimap":
        return
    
    # 같은 키가 반복해서 눌릴 경우 무시, 하지만 None -> None은 허용
    if key == held_key and key is not None:
        # 주기적으로 키를 다시 눌러서 문제 방지 (5초마다)
        if (current_time - last_key_update) > 5.0:
            if held_key:
                keyboard.release(held_key)
                time.sleep(0.05)
                keyboard.press(held_key)
                print(f"[키 입력] {held_key} 키 리프레시 (5초 주기)")
                last_key_update = current_time
        return
    
    # 키 변경 실행
    if held_key:
        keyboard.release(held_key)
        print(f"[키 입력] {held_key} 키 해제 (소스: {source})")
    
    held_key = key
    
    if key:
        keyboard.press(key)
        print(f"[키 입력] {key} 키 누름 (소스: {source})")
    
    last_key_update = current_time
    debounce_counter = 0
    last_held_key = key

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

#########################################
# 게임 창 및 미니맵 관련 함수들
#########################################
def get_game_window():
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "MapleStory Worlds-Mapleland (빅토리아)")
    if hwnd == 0:
        print("🚨 게임 창을 찾을 수 없습니다.")
        return None
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.pointer(rect))
    game_window = {"left": rect.left, "top": rect.top,
                   "width": rect.right - rect.left,
                   "height": rect.bottom - rect.top}
    print(f"[DEBUG] 게임 창 위치: {game_window}")
    return game_window

def validate_region(region, monitor_resolution=(1920,1080)):
    left, top, width, height = region
    left = max(0, left)
    top = max(0, top)
    right = min(left + width, monitor_resolution[0])
    bottom = min(top + height, monitor_resolution[1])
    valid = (left, top, right - left, bottom - top)
    print(f"[DEBUG] 유효한 캡처 영역: {valid}")
    return valid

def select_template(frame, window_name):
    print(f"[INFO] {window_name} 템플릿을 선택하세요! (드래그 후 ENTER, 취소는 'c')")
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = roi
    if w > 0 and h > 0:
        return frame[y:y+h, x:x+w]
    return None

def match_template(frame, template, threshold=0.6):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        t_h, t_w = template.shape[:2]
        return (max_loc[0], max_loc[1], t_w, t_h), max_val
    return None, None

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

def click_callback(event, x, y, flags, param):
    global temp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_points.append((x, y))
        print(f"[DEBUG] 클릭 좌표: ({x}, {y})")

def set_floor_paths(minimap):
    global current_floor, temp_points, floor_paths
    current_floor = 1
    temp_points = []
    floor_paths = {}
    cv2.namedWindow("Set Floor Paths")
    cv2.setMouseCallback("Set Floor Paths", click_callback)
    print("[INFO] 층별 경로 설정: 클릭하여 시작점과 끝점을 지정하세요. 'n'키로 다음 층, ESC로 종료")
    while True:
        temp_img = minimap.copy()
        for pt in temp_points:
            cv2.circle(temp_img, pt, 3, (0, 255, 0), -1)
        if len(temp_points) == 2:
            cv2.line(temp_img, temp_points[0], temp_points[1], (0, 255, 0), 2)
        cv2.putText(temp_img, f"층: {current_floor}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.imshow("Set Floor Paths", temp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            if len(temp_points) == 2:
                floor_paths[current_floor] = (temp_points[0], temp_points[1])
                print(f"[DEBUG] {current_floor}층 경로 설정: {floor_paths[current_floor]}")
                current_floor += 1
                temp_points = []
            else:
                print("[DEBUG] 시작점과 끝점을 모두 설정하세요.")
        elif key == 27:
            if len(temp_points) == 2:
                floor_paths[current_floor] = (temp_points[0], temp_points[1])
                print(f"[DEBUG] {current_floor}층 경로 설정: {floor_paths[current_floor]}")
            break
    cv2.destroyWindow("Set Floor Paths")
    return floor_paths

def point_line_distance(point, line_start, line_end):
    p = np.array(point, dtype=float)
    a = np.array(line_start, dtype=float)
    b = np.array(line_end, dtype=float)
    if np.allclose(a, b):
        return np.linalg.norm(p - a)
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    if t < 0:
        return np.linalg.norm(p - a)
    elif t > 1:
        return np.linalg.norm(p - b)
    projection = a + t * (b - a)
    return np.linalg.norm(p - projection)

# 진입 허용 오차 (픽셀)
route_tolerance = 15

def move_along_route_if_on_path(current_pos, route):
    dist = point_line_distance(current_pos, route[0], route[1])
    print(f"[DEBUG] 현재 위치 {current_pos}와 경로 {route} 사이 거리: {dist:.2f}")
    if dist < route_tolerance:
        # 경로를 리스트로 만들어 move_along_path() 호출
        move_along_path(current_pos, [route[0], route[1]])
    else:
        print("[DEBUG] 캐릭터가 경로에 진입하지 않았습니다.")

def move_along_path(current_pos, path):
    """
    미니맵 좌표계 상에서 드래그한 선(경로)을 따라 이동.
    캐릭터의 x 좌표와 선의 양쪽 끝을 비교하여 진행 방향을 결정하고,
    목표에 도달할 때까지 해당 방향키를 계속 누릅니다.
    """
    global current_direction, threshold, press_tolerance

    if not path or len(path) < 2:
        print("[DEBUG] 경로가 충분하지 않습니다.")
        return

    sorted_path = sorted(path, key=lambda p: p[0])
    left_end = sorted_path[0]
    right_end = sorted_path[-1]
    current_x = current_pos[0]

    # Introduce randomness in the target position
    random_offset = random.randint(-30, 30)
    left_end = (left_end[0] + random_offset, left_end[1])
    right_end = (right_end[0] + random_offset, right_end[1])

    if current_direction is None:
        if current_x <= left_end[0] + threshold:
            current_direction = 'right'
        elif current_x >= right_end[0] - threshold:
            current_direction = 'left'
        else:
            current_direction = 'right' if abs(current_x - right_end[0]) < abs(current_x - left_end[0]) else 'left'
        print(f"[키보드] 초기 진행 방향 결정: {current_direction}")

    target = right_end if current_direction == 'right' else left_end

    if abs(current_x - target[0]) < press_tolerance:
        print(f"[키보드] 현재 위치 {current_pos}가 목표 {target}와 press_tolerance({press_tolerance}) 이내에 있습니다. 방향 전환합니다.")
        # 방향 전환 (키 해제는 스레드에서 처리)
        update_key(None, "minimap")  # 키 일시적 해제
        current_direction = 'left' if current_direction == 'right' else 'right'
        target = right_end if current_direction == 'right' else left_end

    print(f"[키보드] 현재 진행 방향: {current_direction}, 목표: {target}, 현재 x: {current_x}")
    desired_direction = 'right' if current_x < target[0] else 'left'
    print(f"[키보드] 원하는 진행 방향: {desired_direction}")

    # Randomly jump if no monster is detected
    if not monster_detected_in_frame and random.random() < 0.1:  # 10% chance to jump
        keyboard.press('alt')
        time.sleep(0.05)
        keyboard.release('alt')
        print("[키보드] 랜덤 점프")

    # 주가 변경 또는 유지
    update_key(desired_direction, "minimap")

    # Adjust skill delay to 0.8 seconds
    if monster_detected_in_frame:
        time.sleep(1.3)  # Adjusted delay after using skill
        print("[스킬] 딜레이 0.8초 적용")

# 몬스터 감지 정보 업데이트 함수
def update_monster_state(detected, direction=None):
    global monster_direction, shift_pressed, monster_reset_timer, last_monster_time, monster_verify_timer

    current_time = time.time()
    
    if detected:
        monster_direction = direction
        shift_pressed = True
        last_monster_time = current_time
        monster_reset_timer = None
        
        # 몬스터 감지 후 0.3초 후에 재확인 타이머 설정
        if monster_verify_timer is None:
            monster_verify_timer = current_time + 0.1
            print(f"[몬스터] 감지됨 - 방향: {direction}, 0.3초 후 재확인 예정")
    else:
        # 몬스터가 감지되지 않으면 바로 상태를 변경하지 않고
        # 짧은 딜레이 후에 상태를 리셋 (갑작스러운 키 변경 방지)
        if monster_direction is not None and monster_reset_timer is None:
            # 리셋 타이머 시작 (0.3초 딜레이)
            monster_reset_timer = current_time
            # 즉시 Shift 키는 해제 (스킬 사용 중단)
            shift_pressed = False
            print("[몬스터] 감지 해제, Shift 키 즉시 해제, 0.3초 후 이동 모드 복귀")
        
        # 리셋 타이머가 설정되어 있고, 딜레이 시간이 지났으면 리셋
        if monster_reset_timer is not None and (current_time - monster_reset_timer) > 0.3:
            monster_direction = None
            shift_pressed = False
            monster_reset_timer = None
            monster_verify_timer = None
            print("[몬스터] 상태 완전 리셋, 미니맵 이동으로 복귀")

# 몬스터 감지 재확인 함수 (메인 루프에서 호출)
def verify_monster_detection(current_time):
    global monster_verify_timer, monster_direction, shift_pressed
    
    # 재확인 타이머가 설정되어 있고, 시간이 지났으면 재확인
    if monster_verify_timer is not None and current_time >= monster_verify_timer:
        # 몬스터 감지 상태를 재확인 (메인 루프에서 감지 로직 실행 후 결과 확인)
        if not monster_detected_in_frame:
            print("[몬스터] 재확인 결과: 몬스터 없음, 스킬 사용 중단")
            shift_pressed = False  # Shift 키만 해제 (방향키는 유지)
            # 몬스터 방향은 유지 (리셋 타이머에서 처리)
        else:
            print(f"[몬스터] 재확인 결과: 몬스터 계속 감지됨 ({monster_direction})")
        
        # 재확인 타이머 초기화 (다음 감지 때 다시 설정)
        monster_verify_timer = None

#########################################
# mss를 이용한 게임 창 캡처 및 YOLO, 템플릿 매칭 처리
#########################################
def main():
    global character_template, character_template_selected, current_direction
    global monster_detected_in_frame  # 몬스터 감지 상태 변수 추가
    global floor_paths  # floor_paths 전역 변수 추가
    global key_delay, last_key_update  # 키 관련 전역 변수 추가
    global held_key, shift_pressed, monster_direction  # 키 상태 변수 추가
    global monster_reset_timer, monster_verify_timer  # 몬스터 타이머 변수 추가
    global prioritize_minimap, key_thread_running  # 기타 설정 변수 추가
    
    # 디버그 정보를 표시할 검은색 배경 이미지 생성
    debug_window = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 초기화: 모든 키 해제
    keyboard.release('left')
    keyboard.release('right')
    keyboard.release('shift')
    release_key_di(DIK_LEFT)
    release_key_di(DIK_RIGHT)
    release_key_di(DIK_SHIFT)
    print("[메인] 모든 키 초기화 완료")
    
    # 키 입력 스레드 시작
    key_thread = threading.Thread(target=key_pressing_thread)
    key_thread.daemon = True
    key_thread.start()
    print("[메인] 키 입력 스레드 시작됨")
    
    with mss.mss() as sct:
        monitor = get_game_window()
        if monitor is None:
            print("[오류] 게임 창을 찾을 수 없습니다.")
            return
        
        while True:
            current_time = time.time()  # 현재 시간 기록
            monster_detected_in_frame = False  # 매 프레임마다 몬스터 감지 상태 초기화
            
            frame = np.array(sct.grab(monitor))
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            disp_frame = frame.copy()
            
            # 디버그 창 초기화
            debug_window.fill(0)
            
            # --- 메인 화면: YOLO 기반 몬스터 감지 ---
            if character_template_selected:
                char_match, conf = match_template(frame, character_template, threshold=0.6)
                if char_match is not None:
                    cx, cy, cw, ch = char_match
                    cv2.rectangle(disp_frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 2)
                    char_center = (cx + cw//2, cy + ch//2)
                    print(f"[DEBUG] 캐릭터 감지: {char_center} (conf: {conf:.2f})")
                    h_frame, w_frame = frame.shape[:2]
                    rect_x1 = max(0, char_center[0] - 300)
                    rect_x2 = min(w_frame, char_center[0] + 300)
                    rect_y1 = max(0, char_center[1] - 150)
                    rect_y2 = min(h_frame, char_center[1] + ch)
                    cv2.rectangle(disp_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)
                    roi = frame[rect_y1:rect_y2, rect_x1:rect_x2]
                    player_center_roi = (char_center[0] - rect_x1, char_center[1] - rect_y1)
                    
                    # YOLO 추론 (몬스터 감지) - best.pt 모델 사용, 임계값 조정 가능
                    yolo_model = YOLO("runs/detect/train16/weights/best.pt")
                    results = yolo_model(roi, conf=0.9, iou=0.7)
                    candidates = []
                    
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls.item())
                            if cls == 1:
                                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                                monster_center_x = (x_min + x_max) // 2
                                monster_center_y = (y_min + y_max) // 2
                                distance = abs(monster_center_x - player_center_roi[0])
                                candidates.append((distance, monster_center_x, monster_center_y, (x_min, y_min, x_max, y_max)))
                                
                                # 몬스터에 사각형 그리기 (더 두껍고 눈에 띄게)
                                cv2.rectangle(roi, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                                
                                # 몬스터 중심에 원 그리기
                                cv2.circle(roi, (monster_center_x, monster_center_y), 5, (0, 0, 255), -1)
                                
                                # 몬스터 정보 표시
                                cv2.putText(roi, f"Monster", (x_min, y_min-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                # 몬스터와의 거리 표시
                                cv2.putText(roi, f"Dist: {distance:.0f}", (x_min, y_max+15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    if candidates:
                        monster_detected_in_frame = True  # 현재 프레임에서 몬스터 감지됨
                        candidates.sort(key=lambda x: x[0])
                        _, chosen_monster_center_x, chosen_monster_center_y, bbox = candidates[0]
                        direction = "left" if chosen_monster_center_x < player_center_roi[0] else "right"
                        cv2.putText(disp_frame, f"{direction.upper()}", (rect_x1, rect_y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # 캐릭터와 몬스터 사이에 선 그리기
                        cv2.line(roi, 
                                (player_center_roi[0], player_center_roi[1]), 
                                (chosen_monster_center_x, chosen_monster_center_y), 
                                (255, 0, 0), 2)
                        
                        # 몬스터 감지 상태 업데이트 및 이동 명령
                        update_monster_state(True, direction)
                        update_key(direction, "monster")
                    else:
                        # 몬스터가 없으면 감지 상태 업데이트 (키 자동 해제)
                        update_monster_state(False)
                else:
                    # 캐릭터 감지 실패 시
                    update_monster_state(False)
                    update_key(None, "character_lost")
            else:
                # 템플릿 선택 안됨
                update_monster_state(False)
                update_key(None, "no_template")
            
            # 몬스터 감지 재확인 로직 실행
            verify_monster_detection(current_time)
            
            # --- 미니맵 기반 경로 설정 및 이동 ---
            screen_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            minimap_pos = find_minimap(screen_gray)
            if minimap_pos:
                mm_x, mm_y, mm_w, mm_h = minimap_pos
                cv2.rectangle(disp_frame, (mm_x, mm_y), (mm_x+mm_w, mm_y+mm_h), (255, 0, 0), 2)
                minimap = frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w]
                if floor_paths:
                    for floor, route in floor_paths.items():
                        cv2.line(minimap, route[0], route[1], (0, 255, 0), 2)
                        cv2.putText(minimap, f"F{floor}", route[0], cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)
                # 수정: disp_frame에 미니맵을 복사 (screen 대신 disp_frame 사용)
                disp_frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w] = minimap
                char_minimap = find_character_in_minimap(minimap)
                if char_minimap:
                    cx_m, cy_m, cw_m, ch_m = char_minimap
                    abs_cx = mm_x + cx_m
                    abs_cy = mm_y + cy_m
                    cv2.rectangle(disp_frame, (abs_cx, abs_cy), (abs_cx+cw_m, abs_cy+ch_m), (255, 0, 255), 2)
                    current_minimap_pos = (cx_m, cy_m)
                    if floor_paths:
                        # 모든 경로 시각화
                        for floor, route in floor_paths.items():
                            # 경로 그리기
                            route_start, route_end = route
                            cv2.line(disp_frame, 
                                    (mm_x + route_start[0], mm_y + route_start[1]), 
                                    (mm_x + route_end[0], mm_y + route_end[1]), 
                                    (0, 255, 0), 2)
                            # 경로 시작점과 끝점 표시
                            cv2.circle(disp_frame, (mm_x + route_start[0], mm_y + route_start[1]), 5, (0, 0, 255), -1)
                            cv2.circle(disp_frame, (mm_x + route_end[0], mm_y + route_end[1]), 5, (255, 0, 0), -1)
                            
                            # 캐릭터가 경로에 있는지 확인하고 이동 처리
                            move_along_route_if_on_path(current_minimap_pos, route)
                            
                            # 현재 상태 시각화
                            dist = point_line_distance(current_minimap_pos, route[0], route[1])
                            cv2.putText(disp_frame, f"F{floor} dist: {dist:.1f}", (10, 50 + floor*30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                      (0, 255, 0) if dist < route_tolerance else (0, 0, 255), 2)
                
                # 현재 누르고 있는 키 디버그 정보 표시
                key_status = f"현재 키: {held_key if held_key else '없음'}"
                cv2.putText(disp_frame, key_status, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 방향 정보 표시
                direction_status = f"방향: {current_direction if current_direction else '없음'}"
                cv2.putText(disp_frame, direction_status, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                print("[DEBUG] 미니맵 미감지")
                # 미니맵이 감지되지 않으면 모든 키를 해제
                if held_key:
                    print(f"[키보드] 미니맵 미감지로 {held_key} 키 해제")
                    update_key(None, "no_minimap")

            # 디버그 창에 몬스터 감지 상태 표시 개선
            cv2.putText(debug_window, "키보드 상태 디버그", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_window, f"방향키: {held_key if held_key else '없음'}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 255) if held_key else (200, 200, 200), 2)
            cv2.putText(debug_window, f"Shift: {'켜짐' if shift_pressed else '꺼짐'}", (200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 255) if shift_pressed else (200, 200, 200), 2)
            cv2.putText(debug_window, f"미니맵 방향: {current_direction if current_direction else '없음'}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 몬스터 감지 정보 (상태 표시 개선)
            monster_status = f"몬스터: {monster_direction if monster_direction else '미감지'}"
            if monster_verify_timer:
                remaining = max(0, monster_verify_timer - current_time)
                monster_status += f" (재확인: {remaining:.1f}초)"
            elif monster_reset_timer:
                remaining = max(0, 0.3 - (current_time - monster_reset_timer))
                monster_status += f" (해제: {remaining:.1f}초)"
                
            cv2.putText(debug_window, monster_status, (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255) if monster_direction else (200, 200, 200), 2)
            
            # 현재 프레임 몬스터 감지 상태
            frame_status = "현재 프레임: " + ("몬스터 있음" if monster_detected_in_frame else "몬스터 없음")
            cv2.putText(debug_window, frame_status, (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if monster_detected_in_frame else (200, 200, 200), 2)
            
            # 키 업데이트 딜레이 정보
            delay_info = f"키 딜레이: {key_delay:.2f}초, 경과: {time.time() - last_key_update:.2f}초"
            cv2.putText(debug_window, delay_info, (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 2)
            
            # 우선순위 정보
            priority = "미니맵 이동 우선" if prioritize_minimap else "몬스터 감지 우선"
            cv2.putText(debug_window, priority, (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
            
            # 스레드 상태 표시
            thread_status = "스레드 실행 중" if key_thread.is_alive() else "스레드 중지"
            cv2.putText(debug_window, thread_status, (10, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if key_thread.is_alive() else (0, 0, 255), 2)
            
            # 미니맵 정보
            if 'current_minimap_pos' in locals():
                cv2.putText(debug_window, f"캐릭터 위치: {current_minimap_pos}", (10, 280), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if floor_paths:
                    cv2.putText(debug_window, "경로 정보:", (10, 310), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset = 340
                    for floor, route in floor_paths.items():
                        if current_minimap_pos:
                            dist = point_line_distance(current_minimap_pos, route[0], route[1])
                            status = "경로 위" if dist < route_tolerance else "경로 벗어남"
                            color = (0, 255, 0) if dist < route_tolerance else (0, 0, 255)
                            cv2.putText(debug_window, f"F{floor}: {status} (거리: {dist:.1f})", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            y_offset += 30
            
            # 메인 창과 디버그 창 표시
            cv2.imshow("Game Detection", disp_frame)
            cv2.imshow("Keyboard Debug", debug_window)
            key = cv2.waitKey(1) & 0xFF
            
            # 키보드 단축키 처리
            if key == ord('t'):
                temp = select_template(frame, "Select Character Template")
                if temp is not None:
                    character_template = temp
                    character_template_selected = True
                    print("[설정] 캐릭터 템플릿 선택 완료")
            if key == ord('p'):
                if minimap_pos:
                    mm_x, mm_y, mm_w, mm_h = minimap_pos
                    minimap_img = frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w]
                    floor_paths = set_floor_paths(minimap_img)
                    print("[설정] 전체 층 경로 설정 완료:", floor_paths)
            if key == ord('m'):
                prioritize_minimap = not prioritize_minimap
                print(f"[설정] 우선순위 변경: {'미니맵 이동' if prioritize_minimap else '몬스터 감지'} 우선")
            if key == ord('+') or key == ord('='):
                key_delay = min(key_delay + 0.05, 0.5)
                print(f"[설정] 키 딜레이 증가: {key_delay:.2f}초")
            if key == ord('-'):
                key_delay = max(key_delay - 0.05, 0.05)
                print(f"[설정] 키 딜레이 감소: {key_delay:.2f}초")
            if key == ord('c'):
                update_key(None, "manual_clear")
                update_monster_state(False)
                print("[설정] 수동으로 모든 키 해제")
            if key == 27:  # ESC 키
                break
        
        # 프로그램 종료 처리
        print("[메인] 프로그램 종료 중...")
        # 키 해제
        update_key(None, "program_exit")
        update_monster_state(False)
        
        # 스레드 종료
        key_thread_running = False  # 키 입력 스레드 종료 신호
        time.sleep(0.2)  # 스레드가 종료될 시간을 줌
        
        # DirectInput으로 강제 키 해제 (추가적인 안전장치)
        release_key_di(DIK_LEFT)
        release_key_di(DIK_RIGHT)
        release_key_di(DIK_SHIFT)
        
        # 키보드 라이브러리로 강제 키 해제 (추가적인 안전장치)
        keyboard.release('left')
        keyboard.release('right')
        keyboard.release('shift')
        
        cv2.destroyAllWindows()

# GUI 관련 함수 수정
def gui_set_minimap_route():
    """
    미니맵 이동경로 좌표 설정 버튼 클릭 시 실행:
    게임 창 캡처 → 미니맵 좌표 추출 → set_floor_paths 함수 호출
    """
    # 이 작업을 별도 프로세스에서 실행
    import multiprocessing
    p = multiprocessing.Process(target=_process_minimap_route)
    p.start()

def _process_minimap_route():
    # 새 프로세스에서 실행될 함수
    import cv2
    import numpy as np
    import mss
    from tkinter import messagebox, Tk, Button, Label, Frame, BOTH, TOP, BOTTOM, filedialog
    import tkinter as tk
    import os
    import time
    import sys
    
    # 전역 변수
    selected_points = []
    current_floor = 1
    floor_paths = {}
    
    # 게임 창 캡처
    try:
        monitor = get_game_window()
        if monitor is None:
            print("게임 창을 찾을 수 없습니다.")
            with open("error.txt", "w") as f:
                f.write("게임 창을 찾을 수 없습니다.")
            return
        
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            screen_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            minimap_pos = find_minimap(screen_gray)
            
            if minimap_pos is None:
                print("미니맵을 찾지 못했습니다.")
                with open("error.txt", "w") as f:
                    f.write("미니맵을 찾지 못했습니다.")
                return
                
            mm_x, mm_y, mm_w, mm_h = minimap_pos
            minimap_img = frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w].copy()
            
            # 이미지 파일로 저장
            cv2.imwrite("minimap_temp.png", minimap_img)
            print("미니맵 이미지가 저장되었습니다.")
    except Exception as e:
        print(f"이미지 캡처 중 오류 발생: {e}")
        with open("error.txt", "w") as f:
            f.write(f"이미지 캡처 중 오류 발생: {e}")
        return
    
    # Tkinter 창 생성
    root = Tk()
    root.title("미니맵 경로 설정")
    root.geometry("400x500")
    
    # 미니맵 이미지 표시를 위한 캔버스
    pil_img = Image.open("minimap_temp.png")
    tk_img = ImageTk.PhotoImage(pil_img)
    
    # 캔버스 생성
    canvas_frame = Frame(root)
    canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    canvas = tk.Canvas(canvas_frame, width=tk_img.width(), height=tk_img.height())
    canvas.pack()
    
    # 이미지 표시
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    
    # 상태 표시 레이블
    status_frame = Frame(root)
    status_frame.pack(fill=tk.X, padx=10, pady=5)
    
    status_label = Label(status_frame, text=f"현재 층: {current_floor}", font=("Arial", 12))
    status_label.pack(side=tk.LEFT)
    
    points_label = Label(status_frame, text="선택된 점: 0/2", font=("Arial", 12))
    points_label.pack(side=tk.RIGHT)
    
    # 안내 레이블
    instruction_label = Label(root, text="미니맵에서 시작점과 끝점을 클릭하세요", font=("Arial", 10))
    instruction_label.pack(pady=5)
    
    # 마우스 클릭 이벤트 처리
    def on_canvas_click(event):
        nonlocal selected_points
        if len(selected_points) < 2:
            x, y = event.x, event.y
            selected_points.append((x, y))
            
            # 점 표시
            point_id = canvas.create_oval(x-5, y-5, x+5, y+5, fill="green")
            canvas.create_text(x+15, y, text=f"P{len(selected_points)}", fill="green", font=("Arial", 12))
            
            # 상태 업데이트
            points_label.config(text=f"선택된 점: {len(selected_points)}/2")
            print(f"점 {len(selected_points)} 선택: ({x}, {y})")
    
    # 캔버스에 클릭 이벤트 바인딩
    canvas.bind("<Button-1>", on_canvas_click)
    
    # 경로 저장 함수
    def save_path():
        nonlocal current_floor, selected_points
        if len(selected_points) == 2:
            floor_paths[current_floor] = (selected_points[0], selected_points[1])
            print(f"{current_floor}층 경로 설정: {floor_paths[current_floor]}")
            
            # 파일로 저장하여 메인 프로세스와 공유
            with open("floor_paths.txt", "w") as f:
                f.write(str(floor_paths))
            
            messagebox.showinfo("정보", f"{current_floor}층 경로가 저장되었습니다.")
            
            # 다음 층으로 이동
            current_floor += 1
            selected_points = []
            
            # 캔버스 초기화
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            
            # 상태 업데이트
            status_label.config(text=f"현재 층: {current_floor}")
            points_label.config(text="선택된 점: 0/2")
        else:
            messagebox.showwarning("경고", "시작점과 끝점을 모두 설정해주세요.")
    
    # 설정 완료 함수
    def finish_setting():
        if floor_paths:
            # 설정 완료 신호 파일 생성
            with open("paths_ready.txt", "w") as f:
                f.write("ready")
            root.destroy()
        else:
            messagebox.showwarning("경고", "최소한 하나의 층 경로를 설정해주세요.")
    
    # 버튼 프레임
    btn_frame = Frame(root)
    btn_frame.pack(side=BOTTOM, pady=10)
    
    Button(btn_frame, text="경로 저장", command=save_path).pack(side=tk.LEFT, padx=5)
    Button(btn_frame, text="설정 완료", command=finish_setting).pack(side=tk.LEFT, padx=5)
    
    # 메인 루프
    root.mainloop()
    
    # 임시 파일 삭제
    try:
        if os.path.exists("minimap_temp.png"):
            os.remove("minimap_temp.png")
    except:
        pass

def check_and_load_template(root=None):
    """주기적으로 템플릿 파일이 생성되었는지 확인하고 로드"""
    import os
    if os.path.exists("template_ready.txt"):
        try:
            global character_template, character_template_selected
            character_template = cv2.imread("character_template.png")
            if character_template is not None:
                character_template_selected = True
                print("[INFO] 캐릭터 템플릿이 로드되었습니다.")
            os.remove("template_ready.txt")
            os.remove("character_template.png")
        except Exception as e:
            print(f"[ERROR] 템플릿 로드 중 오류: {e}")
    
    # 500ms 후에 다시 확인 (root가 있는 경우에만)
    if root and root.winfo_exists():
        root.after(500, lambda: check_and_load_template(root))

# 경로 로드 함수 추가
def check_and_load_paths(root=None):
    """주기적으로 경로 파일이 생성되었는지 확인하고 로드"""
    import os
    if os.path.exists("paths_ready.txt"):
        try:
            global floor_paths
            with open("floor_paths.txt", "r") as f:
                # 문자열을 딕셔너리로 변환 (eval 사용 - 보안상 위험할 수 있음)
                paths_str = f.read()
                floor_paths = eval(paths_str)
                print("[INFO] 미니맵 경로가 로드되었습니다:", floor_paths)
            os.remove("paths_ready.txt")
            os.remove("floor_paths.txt")
        except Exception as e:
            print(f"[ERROR] 경로 로드 중 오류: {e}")
    
    # 500ms 후에 다시 확인 (root가 있는 경우에만)
    if root and root.winfo_exists():
        root.after(500, lambda: check_and_load_paths(root))

def gui_save_route_coordinates():
    """
    현재 설정된 floor_paths를 파일에 저장
    """
    global floor_paths
    if not floor_paths:
        messagebox.showerror("오류", "저장할 이동경로가 없습니다. 먼저 미니맵 이동경로를 설정해주세요.")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="이동경로 좌표 저장"
    )
    
    if file_path:
        with open(file_path, 'w') as f:
            f.write(str(floor_paths))
        messagebox.showinfo("정보", f"이동경로 좌표가 {file_path}에 저장되었습니다.")

def gui_reset_route_coordinates():
    """
    floor_paths 변수를 초기화
    """
    global floor_paths
    floor_paths = {}
    messagebox.showinfo("정보", "이동경로 좌표가 초기화되었습니다.")

def gui_select_character_template():
    """
    캐릭터 선택 템플릿 매칭 버튼 클릭 시 실행:
    게임 창 캡처 → select_template 호출 → 템플릿 저장
    """
    # 이 작업을 별도 프로세스에서 실행
    import multiprocessing
    p = multiprocessing.Process(target=_process_character_template)
    p.start()

def _process_character_template():
    # 새 프로세스에서 실행될 함수
    import cv2
    import numpy as np
    import mss
    from tkinter import messagebox, Tk, Button, Label, Frame, BOTH, TOP, BOTTOM, Toplevel
    import tkinter as tk
    import os
    import time
    from PIL import Image, ImageTk
    
    # 선택 영역 변수
    start_x, start_y = None, None
    end_x, end_y = None, None
    rect_id = None
    is_selecting = False
    selection_confirmed = False
    
    # 게임 창 캡처
    try:
        monitor = get_game_window()
        if monitor is None:
            print("게임 창을 찾을 수 없습니다.")
            with open("error.txt", "w") as f:
                f.write("게임 창을 찾을 수 없습니다.")
            return
        
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 이미지 파일로 저장
            cv2.imwrite("character_screen_temp.png", frame)
            print("게임 화면이 저장되었습니다.")
    except Exception as e:
        print(f"이미지 캡처 중 오류 발생: {e}")
        with open("error.txt", "w") as f:
            f.write(f"이미지 캡처 중 오류 발생: {e}")
        return
    
    # Tkinter 창 생성
    root = Tk()
    root.title("캐릭터 템플릿 선택")
    root.geometry("800x750")  # 높이를 조금 늘림
    
    # 안내 프레임
    instruction_frame = Frame(root)
    instruction_frame.pack(fill=tk.X, pady=5)
    
    # 안내 레이블
    instruction_label = Label(instruction_frame, text="캐릭터가 있는 영역을 드래그하여 선택하세요", font=("Arial", 12, "bold"))
    instruction_label.pack(pady=5)
    
    # 단계별 안내
    steps_label = Label(instruction_frame, text="1. 드래그로 영역 선택 → 2. 선택 확정 버튼 클릭 → 3. 템플릿 저장 버튼 클릭", 
                        font=("Arial", 10), fg="blue")
    steps_label.pack(pady=2)
    
    # 상태 프레임
    status_frame = Frame(root)
    status_frame.pack(fill=tk.X, pady=5)
    
    status_label = Label(status_frame, text="선택 대기 중...", font=("Arial", 10))
    status_label.pack(side=tk.LEFT, padx=10)
    
    selection_status = Label(status_frame, text="영역 선택: 미완료", font=("Arial", 10), fg="red")
    selection_status.pack(side=tk.RIGHT, padx=10)
    
    # 이미지 로드
    pil_img = Image.open("character_screen_temp.png")
    # 이미지 크기 조정 (창에 맞게)
    screen_width = root.winfo_screenwidth() - 100
    screen_height = root.winfo_screenheight() - 200
    
    img_width, img_height = pil_img.size
    scale = min(screen_width/img_width, screen_height/img_height)
    
    if scale < 1:
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
    
    tk_img = ImageTk.PhotoImage(pil_img)
    
    # 캔버스 생성
    canvas_frame = Frame(root)
    canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    canvas = tk.Canvas(canvas_frame, width=tk_img.width(), height=tk_img.height())
    canvas.pack()
    
    # 이미지 표시
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    
    # 마우스 이벤트 처리
    def on_mouse_down(event):
        nonlocal start_x, start_y, is_selecting, rect_id, selection_confirmed
        
        # 이미 선택이 확정된 경우 무시
        if selection_confirmed:
            return
            
        start_x, start_y = event.x, event.y
        is_selecting = True
        status_label.config(text=f"시작점: ({start_x}, {start_y})")
        
        # 이전 사각형 삭제
        if rect_id:
            canvas.delete(rect_id)
    
    def on_mouse_move(event):
        nonlocal start_x, start_y, end_x, end_y, is_selecting, rect_id, selection_confirmed
        
        # 이미 선택이 확정된 경우 무시
        if selection_confirmed or not is_selecting:
            return
            
        end_x, end_y = event.x, event.y
        
        # 이전 사각형 삭제
        if rect_id:
            canvas.delete(rect_id)
        
        # 새 사각형 그리기
        rect_id = canvas.create_rectangle(start_x, start_y, end_x, end_y, 
                                         outline="green", width=2)
    
    def on_mouse_up(event):
        nonlocal start_x, start_y, end_x, end_y, is_selecting, selection_confirmed
        
        # 이미 선택이 확정된 경우 무시
        if selection_confirmed:
            return
            
        end_x, end_y = event.x, event.y
        is_selecting = False
        
        # 좌표 정렬 (시작점이 항상 왼쪽 위, 끝점이 항상 오른쪽 아래)
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
            
        status_label.config(text=f"선택 영역: ({start_x}, {start_y}) - ({end_x}, {end_y})")
        
        # 영역 크기 확인
        width = abs(end_x - start_x)
        height = abs(end_y - start_y)
        
        if width > 10 and height > 10:
            selection_status.config(text=f"영역 선택: 완료 ({width}x{height})", fg="green")
            confirm_btn.config(state=tk.NORMAL)
            steps_label.config(text="✓ 1. 드래그로 영역 선택 → ✓ 2. 선택 확정 버튼 클릭 → ✓ 3. 템플릿 저장 버튼 클릭", fg="blue")
        else:
            selection_status.config(text="영역 선택: 너무 작음", fg="red")
            confirm_btn.config(state=tk.DISABLED)
    
    # 선택 확정 함수
    def confirm_selection():
        nonlocal selection_confirmed, start_x, start_y, end_x, end_y
        
        if start_x is None or end_x is None or start_y is None or end_y is None:
            messagebox.showwarning("경고", "영역을 선택해주세요.")
            return
        
        if abs(end_x - start_x) < 10 or abs(end_y - start_y) < 10:
            messagebox.showwarning("경고", "너무 작은 영역입니다. 다시 선택해주세요.")
            return
        
        # 선택 확정
        selection_confirmed = True
        
        # UI 업데이트
        selection_status.config(text="영역 선택: 확정됨", fg="blue")
        confirm_btn.config(state=tk.DISABLED)
        reset_btn.config(state=tk.NORMAL)
        save_btn.config(state=tk.NORMAL)
        steps_label.config(text="✓ 1. 드래그로 영역 선택 → ✓ 2. 선택 확정 버튼 클릭 → ✓ 3. 템플릿 저장 버튼 클릭", fg="blue")
        
        # 캔버스에 확정 표시
        canvas.itemconfig(rect_id, outline="blue", width=3)
        
        # 선택 영역 중앙에 "확정" 텍스트 표시
        center_x = (start_x + end_x) // 2
        center_y = (start_y + end_y) // 2
        canvas.create_text(center_x, center_y, text="확정", fill="blue", 
                          font=("Arial", 12, "bold"))
        
        # 미리보기 생성
        try:
            # 원본 이미지에서 선택 영역 추출
            original_img = cv2.imread("character_screen_temp.png")
            img_height, img_width = original_img.shape[:2]
            
            # 캔버스 크기와 원본 이미지 크기의 비율 계산
            width_ratio = img_width / tk_img.width()
            height_ratio = img_height / tk_img.height()
            
            # 원본 이미지 좌표로 변환
            orig_start_x = int(start_x * width_ratio)
            orig_start_y = int(start_y * height_ratio)
            orig_end_x = int(end_x * width_ratio)
            orig_end_y = int(end_y * height_ratio)
            
            # 영역 추출
            template = original_img[orig_start_y:orig_end_y, orig_start_x:orig_end_x]
            
            # 미리보기 저장
            cv2.imwrite("template_preview.png", template)
            
            # 미리보기 표시
            preview_label = Label(status_frame, text="템플릿 미리보기:", font=("Arial", 10))
            preview_label.pack(side=tk.LEFT, padx=10)
            
            # 미리보기 이미지 로드
            preview_img = Image.open("template_preview.png")
            preview_img = preview_img.resize((50, 50), Image.LANCZOS)  # 크기 조정
            preview_tk_img = ImageTk.PhotoImage(preview_img)
            
            # 미리보기 이미지 표시
            preview_img_label = Label(status_frame, image=preview_tk_img)
            preview_img_label.image = preview_tk_img  # 참조 유지
            preview_img_label.pack(side=tk.LEFT)
            
        except Exception as e:
            print(f"미리보기 생성 중 오류: {e}")
    
    # 취소 함수
    def cancel():
        root.destroy()
    
    # 선택 초기화 함수
    def reset_selection():
        nonlocal start_x, start_y, end_x, end_y, rect_id, is_selecting, selection_confirmed
        
        # 변수 초기화
        start_x, start_y = None, None
        end_x, end_y = None, None
        is_selecting = False
        selection_confirmed = False
        
        # UI 초기화
        if rect_id:
            canvas.delete(rect_id)
            rect_id = None
        
        # 모든 캔버스 아이템 삭제 후 이미지만 다시 표시
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        
        # 상태 업데이트
        status_label.config(text="선택 대기 중...")
        selection_status.config(text="영역 선택: 미완료", fg="red")
        steps_label.config(text="1. 드래그로 영역 선택 → 2. 선택 확정 버튼 클릭 → 3. 템플릿 저장 버튼 클릭", fg="blue")
        
        # 버튼 상태 업데이트
        confirm_btn.config(state=tk.DISABLED)
        reset_btn.config(state=tk.DISABLED)
        save_btn.config(state=tk.DISABLED)
        
        # 미리보기 제거
        for widget in status_frame.winfo_children():
            if widget != status_label and widget != selection_status:
                widget.destroy()
    
    # 템플릿 저장 함수
    def save_template():
        nonlocal start_x, start_y, end_x, end_y, selection_confirmed
        
        if not selection_confirmed:
            messagebox.showwarning("경고", "먼저 선택 영역을 확정해주세요.")
            return
        
        try:
            # 원본 이미지에서 선택 영역 추출
            original_img = cv2.imread("character_screen_temp.png")
            img_height, img_width = original_img.shape[:2]
            
            # 캔버스 크기와 원본 이미지 크기의 비율 계산
            width_ratio = img_width / tk_img.width()
            height_ratio = img_height / tk_img.height()
            
            # 원본 이미지 좌표로 변환
            orig_start_x = int(start_x * width_ratio)
            orig_start_y = int(start_y * height_ratio)
            orig_end_x = int(end_x * width_ratio)
            orig_end_y = int(end_y * height_ratio)
            
            # 영역 추출
            template = original_img[orig_start_y:orig_end_y, orig_start_x:orig_end_x]
            
            # 템플릿 저장
            cv2.imwrite("character_template.png", template)
            
            # 완료 신호 파일 생성
            with open("template_ready.txt", "w") as f:
                f.write("ready")
            
            steps_label.config(text="✓ 1. 드래그로 영역 선택 → ✓ 2. 선택 확정 버튼 클릭 → ✓ 3. 템플릿 저장 버튼 클릭", fg="green")
            
            # 템플릿 매칭 테스트
            test_match(original_img, template)
                
            messagebox.showinfo("완료", "캐릭터 템플릿이 저장되었습니다.")
            root.destroy()
            
        except Exception as e:
            messagebox.showerror("오류", f"템플릿 저장 중 오류 발생: {e}")
    
    # 마우스 이벤트 바인딩
    canvas.bind("<Button-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    
    # 버튼 프레임
    btn_frame = Frame(root)
    btn_frame.pack(side=BOTTOM, pady=10)
    
    # 확정 버튼 (초기에는 비활성화)
    confirm_btn = Button(btn_frame, text="선택 확정", command=confirm_selection, 
                        state=tk.DISABLED, width=15, font=("Arial", 10, "bold"))
    confirm_btn.pack(side=tk.LEFT, padx=5)
    
    # 초기화 버튼 (초기에는 비활성화)
    reset_btn = Button(btn_frame, text="선택 초기화", command=reset_selection, 
                      state=tk.DISABLED, width=15)
    reset_btn.pack(side=tk.LEFT, padx=5)
    
    # 저장 버튼 (초기에는 비활성화)
    save_btn = Button(btn_frame, text="템플릿 저장", command=save_template, 
                     state=tk.DISABLED, width=15, font=("Arial", 10, "bold"))
    save_btn.pack(side=tk.LEFT, padx=5)
    
    # 취소 버튼
    Button(btn_frame, text="취소", command=cancel).pack(side=tk.LEFT, padx=5)
    
    # 메인 루프
    root.mainloop()
    
    # 임시 파일 삭제
    try:
        for temp_file in ["character_screen_temp.png", "template_preview.png", "template_match_result.png"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    except:
        pass

# 템플릿 매칭 테스트 함수
def test_match(original_img, template):
    try:
        # 템플릿 매칭 수행
        result = cv2.matchTemplate(original_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 매칭 결과 표시
        h, w = template.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # 결과 이미지에 사각형 그리기
        result_img = original_img.copy()
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
        
        # 매칭 점수 표시
        cv2.putText(result_img, f"Match: {max_val:.2f}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 저장
        cv2.imwrite("template_match_result.png", result_img)
        
        print(f"템플릿 매칭 테스트 완료: 최대 유사도 = {max_val:.4f}")
        
        # 매칭 결과 창 표시
        result_window = Toplevel(root)
        result_window.title("템플릿 매칭 결과")
        result_window.geometry("600x500")
        
        # 결과 이미지 로드
        result_pil_img = Image.open("template_match_result.png")
        
        # 이미지 크기 조정
        screen_width = result_window.winfo_screenwidth() - 100
        screen_height = result_window.winfo_screenheight() - 200
        
        img_width, img_height = result_pil_img.size
        scale = min(screen_width/img_width, screen_height/img_height, 0.5)  # 최대 50%로 제한
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        result_pil_img = result_pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        result_tk_img = ImageTk.PhotoImage(result_pil_img)
        
        # 결과 이미지 표시
        result_label = Label(result_window, image=result_tk_img)
        result_label.image = result_tk_img  # 참조 유지
        result_label.pack(pady=10)
        
        # 매칭 정보 표시
        match_info = Label(result_window, 
                          text=f"매칭 점수: {max_val:.4f}\n위치: {top_left}\n템플릿 크기: {w}x{h}",
                          font=("Arial", 12))
        match_info.pack(pady=10)
        
        # 닫기 버튼
        Button(result_window, text="닫기", command=result_window.destroy).pack(pady=10)
        
    except Exception as e:
        print(f"템플릿 매칭 테스트 중 오류: {e}")

def start_gui():
    """
    GUI 창 시작
    """
    root = tk.Tk()
    root.title("MapleStory Bot GUI")
    
    # 버튼 생성
    btn_set_route = tk.Button(root, text="미니맵 이동경로 좌표 설정", width=30, 
                             command=gui_set_minimap_route)
    btn_set_route.pack(padx=10, pady=5)
    
    btn_select_template = tk.Button(root, text="캐릭터 템플릿 선택", width=30,
                                  command=gui_select_character_template)
    btn_select_template.pack(padx=10, pady=5)
    
    btn_save_route = tk.Button(root, text="이동경로 좌표 저장", width=30, 
                             command=gui_save_route_coordinates)
    btn_save_route.pack(padx=10, pady=5)
    
    btn_reset_route = tk.Button(root, text="이동경로 초기화", width=30, 
                              command=gui_reset_route_coordinates)
    btn_reset_route.pack(padx=10, pady=5)
    
    # 템플릿 및 경로 체크 함수 시작 (root 매개변수 전달)
    check_and_load_template(root)
    check_and_load_paths(root)
    
    root.mainloop()

# 메인 함수 실행
if __name__ == "__main__":
    # 디버그 변수 초기화
    debug_window = None
    debug_minimap = None
    debug_character = None
    
    # 전역 변수 초기화
    last_key_update = time.time()
    debounce_counter = 0
    last_held_key = None
    key_debounce_timer = None
    monster_reset_timer = None
    last_monster_time = 0
    monster_verify_timer = None  # 몬스터 재확인 타이머 추가
    monster_detected_in_frame = False  # 현재 프레임에서 몬스터 감지 여부
    
    # 전역 변수 초기화
    held_key = None  # 누르고 있는 키 저장
    shift_pressed = False  # Shift 키 상태
    monster_direction = None  # 몬스터 방향
    current_direction = None  # 현재 움직이는 방향 (미니맵 기준)
    
    # GUI와 메인 로직을 별도 스레드로 실행
    threading.Thread(target=main, daemon=True).start()
    
    # GUI 시작 (메인 스레드에서 실행)
    start_gui()
