import sys
import time
import subprocess
import multiprocessing
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal
from template_matching import template_matching_process
from yolo_detection import yolo_detection_process
from minimap_detection import find_character_process

class SharedMemoryMonitorThread(QThread):
    update_text = pyqtSignal(str)

    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data

    def run(self):
        last_data = ""
        while True:
            try:
                template_data = self.shared_data["template_matching_results"]
                yolo_data = self.shared_data["yolo_detection_results"]
                minimap_data = self.shared_data["minimap_matching_results"]

                text = (
                    f"📸 템플릿 매칭: {template_data}\n"
                    f"🔍 YOLO 감지: {yolo_data}\n"
                    f"🗺 미니맵 감지: {minimap_data}\n"
                )

                if text != last_data:
                    self.update_text.emit(text)
                    last_data = text

            except Exception as e:
                self.update_text.emit(f"⚠ 오류 발생: {str(e)}")

            time.sleep(0.1)

class DetectionMonitor(QWidget):
    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.initUI()

        self.log_thread = SharedMemoryMonitorThread(self.shared_data)
        self.log_thread.update_text.connect(self.update_log)
        self.log_thread.start()

    def initUI(self):
        self.setWindowTitle("Detection Monitor")
        self.setGeometry(100, 100, 600, 400)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("감지 데이터를 불러오는 중...")

        self.template_button = QPushButton("🔄 템플릿 매칭 시작")
        self.template_button.clicked.connect(self.start_template_matching)

        self.yolo_button = QPushButton("🔄 YOLO 감지 시작")
        self.yolo_button.clicked.connect(self.start_yolo_detection)

        self.minimap_button = QPushButton("🔄 미니맵 감지 시작")
        self.minimap_button.clicked.connect(self.start_minimap_detection)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("📡 감지 결과 모니터링 (공유 메모리)"))
        layout.addWidget(self.log_output)
        layout.addWidget(self.template_button)
        layout.addWidget(self.yolo_button)
        layout.addWidget(self.minimap_button)

        self.setLayout(layout)

    def update_log(self, text):
        self.log_output.setPlainText(text)

    def start_template_matching(self):
        process = multiprocessing.Process(target=template_matching_process, args=(self.shared_data,))
        process.start()

    def start_yolo_detection(self):
        process = multiprocessing.Process(target=yolo_detection_process, args=(self.shared_data,))
        process.start()

    def start_minimap_detection(self):
        process = multiprocessing.Process(target=find_character_process, args=(self.shared_data,))
        process.start()

def run_gui(shared_data):
    app = QApplication(sys.argv)
    window = DetectionMonitor(shared_data)
    window.show()
    sys.exit(app.exec_())