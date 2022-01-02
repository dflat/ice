# data structures for network transmission
# representing game state and controller input.

class ButtonState:
    JUMP = 0x01

class Record:
    __slots__ = ('roll', 'pitch', 'yaw', 'buttons', 't')
    fmt = 'bbbb'

    def __init__(self, roll, pitch, yaw, buttons=0x00, t=0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.buttons = buttons
        self.t = t

    def jump_pressed(self):
        return True if self.buttons & ButtonState.JUMP else False

    def button_pressed(self, button):
        '''
        function to test if a button was pressed.
        self.button is a byte, with each bit representing 8 on/off bools
        '''
        return True if self.buttons & button else False

    def __repr__(self):
        return f'Record({self.roll:.2f}, {self.pitch:.2f}, {self.yaw:.2f}'\
                f', t={self.t})'

