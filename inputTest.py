import pyautogui
import time
import keyboard  # keyboard 모듈을 사용하여 종료 키를 감지

def main():
    print("Ctrl 키를 1초에 한 번씩 입력합니다. 종료하려면 'Esc' 키를 누르세요.")

    try:
        while True:
            # Ctrl 키 입력
            pyautogui.keyDown('ctrl')
            time.sleep(0.1)  # 짧은 시간 동안 키를 누르고 있도록 설정
            pyautogui.keyUp('ctrl')

            # 1초 대기
            time.sleep(1)

            # Esc 키가 눌리면 종료
            if keyboard.is_pressed('esc'):
                print("종료합니다.")
                break

    except KeyboardInterrupt:
        print("프로그램이 중단되었습니다.")

if __name__ == "__main__":
    main()
