import json
import os

class ConfigManager:
    def __init__(self, config_path="keybindings.json"):
        self.config_path = config_path
        self.user_keybindings = {}

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.user_keybindings = json.load(f)
            print(f"[ConfigManager] Loaded keybindings from {self.config_path}")
        else:
            # 파일이 없으면 기본값 설정
            self.user_keybindings = {
                "skill_1": "1",
                "skill_2": "2",
                "buff_1": "3",
                "hp_potion": "f1",
                "jump": "space"
            }
            print("[ConfigManager] No config file found. Using default keybindings.")

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.user_keybindings, f, indent=4, ensure_ascii=False)
        print(f"[ConfigManager] Saved keybindings to {self.config_path}")