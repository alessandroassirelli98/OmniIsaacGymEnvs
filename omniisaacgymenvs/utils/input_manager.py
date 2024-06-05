import numpy as np
from pynput import keyboard
import ctypes
import threading

class KeyboardManager():
    def __init__(self, hand_coord):
        self.action = hand_coord
        self.hand_close_idx = len(self.action)-1
        self.kill = False
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def on_press(self, key):
        if (key == keyboard.Key.up):
            self.action[0] = 1
        if (key == keyboard.Key.down):
            self.action[0] = -1
        if (key == keyboard.Key.left):
            self.action[1] = 1
        if (key == keyboard.Key.right):                                  
            self.action[1] = -1
        if (key == keyboard.Key.page_up):
            self.action[2] = 1
        if (key == keyboard.Key.page_down):                                  
            self.action[2] = -1
        if (key == keyboard.Key.shift_l):
            self.action[3] = 1
        if (key == keyboard.Key.shift_r):                                  
            self.action[3] = -1

        if (key == keyboard.Key.space):
            self.action[self.hand_close_idx] = 1

        if (key == keyboard.Key.esc):
            self.kill = True
    
    def on_release(self, key):
        if (key == keyboard.Key.up or keyboard.Key.down):
            self.action[0] = 0
        if (key == keyboard.Key.left or key == keyboard.Key.right):
            self.action[1] = 0
        if (key == keyboard.Key.page_up or key == keyboard.Key.page_down):
            self.action[2] = 0
        if (key == keyboard.Key.shift_l or key ==keyboard.Key.shift_r):                                  
            self.action[3] = 0

        if (key == keyboard.Key.space):
            self.action[self.hand_close_idx] = 0


class SpaceMouseManager():
    def __init__(self, hand_coord):
        self.action = hand_coord
        self.hand_close_idx = len(self.action) - 1
        self.kill = False

        # Create and start the thread
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        import pyspacemouse
        try:
            success = pyspacemouse.open()
            if success:
                while 1:
                    state = pyspacemouse.read()
                    self.action[0] = state.y
                    self.action[1] = state.x
                    self.action[2] = state.z
                    self.action[3] = state.roll
                    self.action[4] = state.pitch
                    self.action[5] = state.yaw
                    self.action[6] = state.buttons[0]
                    self.kill = bool(state.buttons[1])
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.kill = True
        self.thread.join()


if __name__ == "__main__":
    import torch
    
    action = torch.zeros(7, dtype=torch.float32)
    sm = SpaceMouseManager(action)
    print(action)
