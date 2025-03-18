import pydirectinput
import time
import keyboard
import threading
from config_manager import ConfigManager
from key_input_manager import KeyInputManager

def check_esc_key(should_stop):
    while not should_stop.is_set():
        if keyboard.is_pressed('esc'):
            print("종료합니다.")
            should_stop.set()
            break
        time.sleep(0.1)

def main():
    # ConfigManager와 KeyInputManager 초기화
    config_manager = ConfigManager()
    config_manager.load_config()
    key_input_manager = KeyInputManager(config_manager)

    print("공격 키 바인딩:", config_manager.user_keybindings.get('attack', 'Not set'))
    print("\n공격 키 입력을 시작합니다. 종료하려면 'Esc' 키를 누르세요.")
    
    # pydirectinput 초기화
    pydirectinput.PAUSE = 0.1  # 키 입력 간 딜레이 설정

    # ESC 키 감지를 위한 이벤트 플래그
    should_stop = threading.Event()
    
    # ESC 키 감지 스레드 시작
    esc_thread = threading.Thread(target=check_esc_key, args=(should_stop,))
    esc_thread.daemon = True  # 메인 스레드가 종료되면 함께 종료
    esc_thread.start()

    try:
        while not should_stop.is_set():
            # 공격 키만 입력
            key_input_manager.jump()
            time.sleep(1)  # 1초 대기

    except KeyboardInterrupt:
        print("프로그램이 중단되었습니다.")
    finally:
        # 프로그램 종료 시 모든 키를 떼기
        pydirectinput.keyUp('ctrl')
        pydirectinput.keyUp('alt')
        pydirectinput.keyUp('shift')

if __name__ == "__main__":
    main()
