import pydirectinput
import time
import keyboard
import threading
from config_manager import ConfigManager
from key_input_manager import KeyInputManager
from states.base_state import BaseState

class AttackState(BaseState):
    def __init__(self, key_input_manager):
        self.key_input_manager = key_input_manager
        self.last_attack_time = 0
        self.attack_interval = 1.0  # 1초 간격

    def update(self):
        current_time = time.time()
        if current_time - self.last_attack_time >= self.attack_interval:
            self.key_input_manager.attack()
            self.last_attack_time = current_time

class JumpState(BaseState):
    def __init__(self, key_input_manager):
        self.key_input_manager = key_input_manager
        self.last_jump_time = 0
        self.jump_interval = 1.0  # 1초 간격

    def update(self):
        current_time = time.time()
        if current_time - self.last_jump_time >= self.jump_interval:
            self.key_input_manager.jump()
            self.last_jump_time = current_time

class StateManager:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
        self.key_input_manager = KeyInputManager(self.config_manager)
        self.current_state = None
        self.should_stop = threading.Event()
        
    def set_state(self, state):
        if self.current_state:
            self.current_state.on_exit()
        self.current_state = state
        self.current_state.on_enter()
        
    def update(self):
        if self.current_state:
            self.current_state.update()

def check_esc_key(should_stop):
    while not should_stop.is_set():
        if keyboard.is_pressed('esc'):
            print("종료합니다.")
            should_stop.set()
            break
        time.sleep(0.1)

def main():
    # pydirectinput 초기화
    pydirectinput.PAUSE = 0.1

    # 상태 관리자 초기화
    state_manager = StateManager()
    
    # 초기 상태 설정 (공격 상태)
    state_manager.set_state(AttackState(state_manager.key_input_manager))
    
    print("현재 상태: 공격")
    print("공격 키 바인딩:", state_manager.config_manager.user_keybindings.get('attack', 'Not set'))
    print("\n실행 중입니다. 종료하려면 'Esc' 키를 누르세요.")
    
    # ESC 키 감지 스레드 시작
    esc_thread = threading.Thread(target=check_esc_key, args=(state_manager.should_stop,))
    esc_thread.daemon = True
    esc_thread.start()

    try:
        while not state_manager.should_stop.is_set():
            state_manager.update()
            time.sleep(0.1)  # CPU 사용량 감소를 위한 짧은 대기

    except KeyboardInterrupt:
        print("프로그램이 중단되었습니다.")
    finally:
        # 프로그램 종료 시 모든 키를 떼기
        pydirectinput.keyUp('ctrl')
        pydirectinput.keyUp('alt')
        pydirectinput.keyUp('shift')

if __name__ == "__main__":
    main()
