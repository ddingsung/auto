import cv2
import numpy as np
import mss
import os
import glob
import time

def template_matching_process(shared_data):
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
                threshold = 0.5  # 임계값 (필요에 따라 조절)
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
            # 공유 메모리에 감지 결과 저장장
            shared_data["template_matching_results"] = detected_centers
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()