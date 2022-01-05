# data structures for network transmission
# representing game state and controller input.

class ButtonState:
    JUMP = 0x01
    SLOWMO = 0x02
    SLOWMO_EXIT = 0x04

class Record:
    __slots__ = ('seq_no', 'roll', 'pitch', 'yaw', 'buttons', 'dy', 't')
    fmt = 'Ibbbbb'

    def __init__(self, seq_no, roll, pitch, yaw, buttons=0x00, dy=0, t=0):
        self.seq_no = seq_no
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.buttons = buttons
        self.dy = dy
        self.t = t

    def jump_pressed(self):
        return True if self.buttons & ButtonState.JUMP else False

    def slow_mo_pressed(self):
        return True if self.buttons & ButtonState.SLOWMO else False

    def slow_mo_exit_pressed(self):
        return True if self.buttons & ButtonState.SLOWMO_EXIT else False

    def button_pressed(self, button):
        '''
        function to test if a button was pressed.
        self.button is a byte, with each bit representing 8 on/off bools
        '''
        return True if self.buttons & button else False

    def __repr__(self):
        buttons_pressed = []
        if self.jump_pressed():
            buttons_pressed.append('jump')
        if self.slow_mo_pressed():
            buttons_pressed.append('slow-mo')
        btns = ', '.join(buttons_pressed)
        return f'Record(seq_no={self.seq_no}, roll={self.roll}, pitch={self.pitch}, '\
               f'dy={self.dy}, buttons=[{btns}])'

