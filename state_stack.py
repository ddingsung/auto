# state_stack.py

from typing import List
from states.base_state import BaseState

class StateStack:
    def __init__(self):
        self.stack: List[BaseState] = []

    def push(self, state: BaseState):
        # 현재 활성 상태를 일시정지
        if self.stack:
            self.stack[-1].on_pause()

        self.stack.append(state)
        state.on_enter()

    def pop(self):
        if self.stack:
            top_state = self.stack.pop()
            top_state.on_exit()
            # 스택이 비어있지 않다면 직전 상태를 재개
            if self.stack:
                self.stack[-1].on_resume()

    def update(self):
        # 스택 최상단 상태만 업데이트(실행)
        if self.stack:
            self.stack[-1].update()
