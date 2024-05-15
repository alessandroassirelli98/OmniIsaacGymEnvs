import numpy as np
from pynput import keyboard
import ctypes
import time

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
        spnav = ctypes.CDLL('libspnav.so')

        # Function prototypes
        spnav.spnav_open.restype = ctypes.c_int
        spnav.spnav_close.restype = ctypes.c_int
        self.action = hand_coord
        self.hand_close_idx = len(self.action)-1
        self.kill = False
        
        last_time = time.time()
        last_translation = [0, 0, 0]
        last_rotation = [0, 0, 0]

        if spnav.spnav_open() == -1:
            print("Failed to connect to the space navigator daemon")
            return
        
        while True:
            event = SpnavEvent()
            if spnav.spnav_poll_event(ctypes.byref(event)) > 0:
                current_time = time.time()
                dt = current_time - last_time

                dx = event.motion.x - last_translation[0]
                dy = event.motion.y - last_translation[1]
                dz = event.motion.z - last_translation[2]

                drx = event.motion.rx - last_rotation[0]
                dry = event.motion.ry - last_rotation[1]
                drz = event.motion.rz - last_rotation[2]

                vx = dx / dt
                vy = dy / dt
                vz = dz / dt

                vrx = drx / dt
                vry = dry / dt
                vrz = drz / dt

                print(f"Velocity: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
                print(f"Rotational Velocity: vrx={vrx:.2f}, vry={vry:.2f}, vrz={vrz:.2f}")

                last_translation = [event.motion.x, event.motion.y, event.motion.z]
                last_rotation = [event.motion.rx, event.motion.ry, event.motion.rz]
                last_time = current_time


if __name__ == "__main__":
    sm = SpaceMouseManager([0])
