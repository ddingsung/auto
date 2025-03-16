import multiprocessing
from gui import run_gui  # Import the run_gui function from gui.py

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    # ✅ 공유 메모리 생성 (Queue 대신 dict 사용)
    manager = multiprocessing.Manager()
    shared_data = manager.dict()

    # 초기값 설정
    shared_data["template_matching_results"] = []
    shared_data["yolo_detection_results"] = []
    shared_data["minimap_matching_results"] = []

    # GUI 실행
    run_gui(shared_data)
