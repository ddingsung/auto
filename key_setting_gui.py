from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeyEvent
import sys
from config_manager import ConfigManager
from key_input_manager import KeyInputManager

class KeySettingButton(QPushButton):
    def __init__(self, action_name, current_key, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        self.current_key = current_key
        self.waiting_for_key = False
        self.update_text()
        self.clicked.connect(self.start_key_capture)
        
    def update_text(self):
        self.setText(f"{self.action_name}: {self.current_key}")
        
    def start_key_capture(self):
        self.waiting_for_key = True
        self.setText(f"{self.action_name}: Press any key...")
        self.setStyleSheet("background-color: #ffeb3b;")
        
    def keyPressEvent(self, event: QKeyEvent):
        if self.waiting_for_key:
            # Get modifier keys
            modifiers = []
            if event.modifiers() & Qt.ShiftModifier:
                modifiers.append('SHIFT')
            if event.modifiers() & Qt.ControlModifier:
                modifiers.append('CTRL')
            if event.modifiers() & Qt.AltModifier:
                modifiers.append('ALT')
            if event.modifiers() & Qt.MetaModifier:
                modifiers.append('META')
            
            # If only modifier keys are pressed, use the first modifier
            if modifiers and not event.text():
                key_text = modifiers[0]
            else:
                # Get the key text
                key_text = event.text().upper()
                if not key_text or key_text == ' ':
                    # Handle special keys
                    if event.key() == Qt.Key_Space:
                        key_text = 'SPACEBAR'
                    elif event.key() == Qt.Key_Return:
                        key_text = 'ENTER'
                    elif event.key() == Qt.Key_Escape:
                        key_text = 'ESC'
                    elif event.key() == Qt.Key_Backspace:
                        key_text = 'BACKSPACE'
                    elif event.key() == Qt.Key_Tab:
                        key_text = 'TAB'
                    elif event.key() == Qt.Key_Delete:
                        key_text = 'DELETE'
                    elif event.key() == Qt.Key_Insert:
                        key_text = 'INSERT'
                    elif event.key() == Qt.Key_Home:
                        key_text = 'HOME'
                    elif event.key() == Qt.Key_End:
                        key_text = 'END'
                    elif event.key() == Qt.Key_PageUp:
                        key_text = 'PAGEUP'
                    elif event.key() == Qt.Key_PageDown:
                        key_text = 'PAGEDOWN'
                    elif event.key() == Qt.Key_Left:
                        key_text = 'LEFT'
                    elif event.key() == Qt.Key_Right:
                        key_text = 'RIGHT'
                    elif event.key() == Qt.Key_Up:
                        key_text = 'UP'
                    elif event.key() == Qt.Key_Down:
                        key_text = 'DOWN'
                    elif event.key() == Qt.Key_F1:
                        key_text = 'F1'
                    elif event.key() == Qt.Key_F2:
                        key_text = 'F2'
                    elif event.key() == Qt.Key_F3:
                        key_text = 'F3'
                    elif event.key() == Qt.Key_F4:
                        key_text = 'F4'
                    elif event.key() == Qt.Key_F5:
                        key_text = 'F5'
                    elif event.key() == Qt.Key_F6:
                        key_text = 'F6'
                    elif event.key() == Qt.Key_F7:
                        key_text = 'F7'
                    elif event.key() == Qt.Key_F8:
                        key_text = 'F8'
                    elif event.key() == Qt.Key_F9:
                        key_text = 'F9'
                    elif event.key() == Qt.Key_F10:
                        key_text = 'F10'
                    elif event.key() == Qt.Key_F11:
                        key_text = 'F11'
                    elif event.key() == Qt.Key_F12:
                        key_text = 'F12'
                    else:
                        # Convert Qt key code to actual key name
                        key = event.key()
                        if Qt.Key_A <= key <= Qt.Key_Z:
                            key_text = chr(key)
                        elif Qt.Key_0 <= key <= Qt.Key_9:
                            key_text = chr(key)
                        else:
                            # For other keys, use a more readable format
                            key_text = f"KEY_{key}"
            
            # Combine modifiers and key only if we have both
            if modifiers and key_text and key_text not in modifiers:
                key_text = '+'.join(modifiers + [key_text])
                
            self.current_key = key_text
            self.waiting_for_key = False
            self.setStyleSheet("")
            self.update_text()
            
            # Find the main window and call its method
            main_window = self.window()
            if isinstance(main_window, KeySettingWindow):
                main_window.key_set_changed(self.action_name, key_text)

class KeySettingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
        self.key_input_manager = KeyInputManager(self.config_manager)
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Key Settings')
        self.setGeometry(100, 100, 400, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create key setting buttons
        self.key_buttons = {}
        
        # Define actions and their display names
        actions = {
            'skill_1': '스킬 1',
            'skill_2': '스킬 2',
            'buff_1': '버프 1',
            'buff_2': '버프 2',
            'buff_3': '버프 3',
            'buff_4': '버프 4',
            'buff_5': '버프 5',
            'pet_potion': '펫 물약',
            'hp_potion': 'HP 물약',
            'mp_potion': 'MP 물약',
            'jump': '점프',
            'attack': '공격'
        }
        
        # Create buttons for each action
        for action, display_name in actions.items():
            current_key = self.config_manager.user_keybindings.get(action, '')
            button = KeySettingButton(display_name, current_key, self)
            self.key_buttons[action] = button
            layout.addWidget(button)
        
        # Add save button
        save_button = QPushButton('설정 저장', self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        # Add status label
        self.status_label = QLabel('', self)
        layout.addWidget(self.status_label)
        
    def key_set_changed(self, action_name, new_key):
        # Find the action from the display name
        for action, display_name in self.key_buttons.items():
            if display_name.action_name == action_name:
                self.config_manager.user_keybindings[action] = new_key
                self.status_label.setText(f'키 설정 변경됨: {action_name} -> {new_key}')
                break
                
    def save_settings(self):
        try:
            self.config_manager.save_config()
            self.status_label.setText('설정이 저장되었습니다.')
            QMessageBox.information(self, '성공', '키 설정이 저장되었습니다.')
        except Exception as e:
            QMessageBox.critical(self, '오류', f'설정 저장 중 오류가 발생했습니다: {str(e)}')
            
    def keyPressEvent(self, event: QKeyEvent):
        # Forward key events to the focused button
        focused_button = self.focusWidget()
        if isinstance(focused_button, KeySettingButton):
            focused_button.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    window = KeySettingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
