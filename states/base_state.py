# states/base_state.py

class BaseState:
    def on_enter(self):
        """상태에 진입할 때 딱 한 번 실행"""
        pass

    def on_pause(self):
        """다른 상태가 push될 때 일시정지"""
        pass

    def on_resume(self):
        """스택에서 pop되어 다시 돌아올 때 재개"""
        pass

    def on_exit(self):
        """상태에서 완전히 벗어날 때 한 번 실행"""
        pass

    def update(self):
        """매 프레임 or 매 루프마다 실행"""
        pass
