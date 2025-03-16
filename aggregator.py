import time

def aggregator(shared_data):
    """
    공유 큐에 들어오는 감지 결과들을 읽어서 출력합니다.
    """
    while True:
        template_matching_results = shared_data["template_matching_results"]
        yolo_detection_results = shared_data["yolo_detection_results"]
        minimap_matching_results = shared_data["minimap_matching_results"]
        print(f"Template Matching Results: {template_matching_results}")
        print(f"YOLO Detection Results: {yolo_detection_results}")
        print(f"Minimap Matching Results: {minimap_matching_results}")
        time.sleep(0.05)