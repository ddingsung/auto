import pydirectinput
import time

class KeyInputManager:
    def __init__(self, config_manager):
        """
        config_manager.user_keybindings 딕셔너리에 
        {'skill_1': '1', 'hp_potion': 'f1', ... } 형태로 저장돼 있다고 가정.
        """
        self.config_manager = config_manager

    def press_key(self, action_name):
        """
        action_name = 'skill_1', 'hp_potion', 'jump' 등
        에 해당하는 실제 키값을 찾아 pydirectinput.press()로 누른다.
        """
        key_bindings = self.config_manager.user_keybindings
        if action_name not in key_bindings:
            print(f"[KeyInputManager] No key bound for action '{action_name}'")
            return

        key_to_press = key_bindings[action_name]
        print(f"[KeyInputManager] Pressing {action_name} ({key_to_press})")
        pydirectinput.press(key_to_press)
        # 필요 시 time.sleep()으로 연타 간격 조절 가능

    def hold_key(self, action_name, duration=0.5):
        """
        특정 키를 일정 시간(duration) 동안 누르고 있는 예시.
        """
        key_bindings = self.config_manager.user_keybindings
        if action_name not in key_bindings:
            print(f"[KeyInputManager] No key bound for action '{action_name}'")
            return

        key_to_press = key_bindings[action_name]
        print(f"[KeyInputManager] Holding {action_name} ({key_to_press}) for {duration} sec")
        pydirectinput.keyDown(key_to_press)
        time.sleep(duration)
        pydirectinput.keyUp(key_to_press)

    # 자주 쓰이는 기능 직접 래핑
    
    def use_hp_potion(self):
        self.press_key("hp_potion")

    def jump(self):
        self.press_key("jump")

    def attack(self):
        self.press_key("attack")

    def mp_potion(self):
        self.press_key("mp_potion")

    def buff_1(self):
        self.press_key("buff_1")

    def buff_2(self):
        self.press_key("buff_2")

    def buff_3(self):
        self.press_key("buff_3")

    def buff_4(self):
        self.press_key("buff_4")

    def buff_5(self):
        self.press_key("buff_5")

    def pet_potion(self):
        self.press_key("pet_potion")

    def skill_1(self):
        self.press_key("skill_1")

    def skill_2(self):
        self.press_key("skill_2")

        
        
