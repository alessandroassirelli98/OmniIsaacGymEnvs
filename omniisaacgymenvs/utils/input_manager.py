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
        if (key == keyboard.Key.shift_l or key ==keyboard.Key.shift_l):                                  
            self.action[3] = 0

        if (key == keyboard.Key.space):
            self.action[self.hand_close_idx] = 0

# Event structure
class SpnavEventMotion(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                    ("device", ctypes.c_int),
                    ("x", ctypes.c_int),
                    ("y", ctypes.c_int),
                    ("z", ctypes.c_int),
                    ("rx", ctypes.c_int),
                    ("ry", ctypes.c_int),
                    ("rz", ctypes.c_int),
                    ("period", ctypes.c_uint)]

class SpnavEventButton(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("device", ctypes.c_int),
                ("press", ctypes.c_int),
                ("bnum", ctypes.c_int)]

class SpnavEvent(ctypes.Union):
    _fields_ = [("type", ctypes.c_int),
                ("motion", SpnavEventMotion),
                ("button", SpnavEventButton)]
class SpaceMouseManager():
    def __init__(self, hand_coord):
        # Load the libspnav library
        self.spnav = ctypes.CDLL('libspnav.so')

        # Function prototypes
        self.spnav.spnav_open.restype = ctypes.c_int
        self.spnav.spnav_close.restype = ctypes.c_int

        self.action = hand_coord
        self.hand_close_idx = len(self.action) - 1
        self.kill = False

        if self.spnav.spnav_open() == -1:
            print("Failed to connect to the space navigator daemon")
            return

        # Create and start the thread
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        try:
            while not self.kill:
                event = SpnavEvent()
                if self.spnav.spnav_poll_event(ctypes.byref(event)) > 0:
                    if event.type == 1:  # Motion event
                        self.set_action(event.motion.x, 15, 0)
                        self.set_action(event.motion.y, 15, 1)
                        self.set_action(event.motion.z, 15, 2)

                    elif event.type == 2:  # Button event
                        # print(f"Button {event.button.bnum} {'pressed' if event.button.press else 'released'}")
                        if (event.button.press):
                            self.action[self.hand_close_idx] = 1
                        else:
                            self.action[self.hand_close_idx] = 0

        except KeyboardInterrupt:
            pass
        finally:
            self.spnav.spnav_close()  # Close the connection to the SpaceMouse

    def stop(self):
        self.kill = True
        self.thread.join()

    def set_action(self, axis, threshold, idx):
        if axis > threshold:
            self.action[idx] = 1
        elif axis < -threshold:
            self.action[idx] = -1
        else:
            self.action[idx] = 0


if __name__ == "__main__":
    import torch
    
    action = torch.zeros(5, dtype=torch.int16)
    sm = SpaceMouseManager(action)
