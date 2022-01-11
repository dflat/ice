import numpy as np
import wave
import io
import os
import time
import threading
import pygame.locals
from utils import wave_chop
#from . import config

def interp(arr, stretch=2):
    l, r = arr.T
    n = len(l)
    n_stretch_samples = int(n*stretch)
    t = np.linspace(0,1,n)
    t_stretch = np.linspace(0,1, n_stretch_samples)
    l_stretch = np.interp(t_stretch, t, l)
    r_stretch = np.interp(t_stretch, t, r)
    stretched_arr = np.column_stack((l_stretch, r_stretch))
    return stretched_arr.astype('int16')

def repeat_stretch(arr, stretch=2):
    l,r = arr.T
    ls = np.repeat(l, stretch, axis=0)
    rs = np.repeat(r, stretch, axis=0)
    return np.column_stack((ls,rs))

def stream_wav(chunks:'List[path]'):
    """
    Stream a wav file in chunks, using pygame's mixer's queue
    """
    cache = { }
    def queue_up(chunk: 'wavefile'):
        pygame.mixer.music.queue(chunk, namehint='wav')
    def get_chunk_len(i):
        dur = cache.get((i,'dur'))
        if not dur:
            with wave.open(path) as w:
                chans, width, fr, n_frames, *rest = w.getparams()
                dur = n_frames / fr
            cache[(i,'dur')] = dur
        return dur
    def get_chunk(i: int):
        f = cache.get(i)
        if not f:
            f = open(chunks[i], 'rb')
            cache[i] = f
        return f
    def arr_to_buf(arr):
        return io.BytesIO(arr.tobytes())
    def clear_cache():
        for k, v in cache.items():
            if hasattr(v, 'close'):
                v.close()
    n = len(chunks)
    i = 0
    playhead = 0
    chunk = get_chunk(i)
    chunk_len = get_chunk_len(i)
    pygame.mixer.music.load(chunk, namehint='wav')
    pygame.mixer.music.play()
    start_t = time.time()
    last_play_time = 0
    last_q_time = 0
    time_til_next_q = chunk_len / 2
    next_q_time = playhead + time_til_next_q
    while True:
        playhead = time.time() - start_t
        if playhead - next_q_time > 0:
            i += 1
            next_chunk = get_chunk(i % n)
            next_chunk_len = get_chunk_len(i % n)
            queue_up(next_chunk)
            
            last_q_time = playhead
            elapsed = pygame.mixer.music.get_pos() / 1000
            time_til_next_play = chunk_len - elapsed 
            time_til_next_q = time_til_next_play + next_chunk_len/2
            next_q_time = playhead + time_til_next_q
            #print(f'queued up a {next_chunk_len:.1f}s chunk ({i}) '
            #      f'at t={playhead}')
            
            chunk_len = next_chunk_len
        if i == (n-1):
            break 
        delay = next_q_time - playhead 
        if delay > 0:
            time.sleep(delay)
    clear_cache()

def get_chunk_len(path):
    with wave.open(path) as w:
        chans, width, fr, n_frames, *rest = w.getparams()
        return n_frames / fr

def wav_to_arr(path):
    with wave.open(path) as w:
        buf = w.readframes(w.getnframes())
        a = np.frombuffer(buf, dtype='int16')
        return a.reshape(w.getnframes(), w.getnchannels())

class WaveChunk:
    __slots__ = ('path','arr','stretched_arr','snd','stretched_snd','dur','active_snd')
    def __init__(self, *args):
        for i, attr in enumerate(self.__slots__[:-1]):
            setattr(self, attr, args[i])
        self.active_snd = self.snd
        
class WaveStream:
    ROOT = os.path.join('assets','sound','music')
    def __init__(self, src_path, *, segment_dur=0.1, stretch=2,
                 aim_for_even_chunks=False, overwrite=False):
        name, ext = src_path.rsplit('.')
        assert(ext == 'wav')
        self.segment_dur = segment_dur
        self.overwrite = overwrite
        self.dir = name + '_chunks'
        self.dirpath = os.path.join(self.ROOT, self.dir)
        self.src_path = os.path.join(self.ROOT, src_path)
        pygame.mixer.set_reserved(1)
        self.channel = pygame.mixer.Channel(0)
        self.chop(aim_for_even_chunks=aim_for_even_chunks)
        self.stretch = stretch
        self.make_chunks(stretch=stretch)
        self.channel.set_endevent(pygame.locals.USEREVENT)
        self.chunk_end_event = self.channel.get_endevent() 
        self.index = 0
        self.n_chunks = len(self.chunks.get(1))
        self.rate = 1
    
    def shutdown(self):
        self._remote.set()

    def play_threaded(self):
        self._remote = threading.Event()
        t = threading.Thread(target=self._play_threaded)
        t.start()
        
    def _play_threaded(self):
        delay = self.segment_dur / 3 #2
        self.play()
        misses = 0
        while not self._remote.is_set():
            if not self.channel.get_queue(): # no chunk is queue'd up
                self.queue_next()
                #print('misses:',misses)
                misses = 0
            misses += 1
            time.sleep(delay)
        print('music stream was shutdown.')

    def play(self):
        """
        Load the first two chunks (one will play immediately,
        and the second will be queued).
        """
        self._streaming = True
        self.channel.play(self.chunks[self.rate][self.index])
        self.queue_next()

    def set_rate(self, rate: int):
        if rate not in self.chunks:
            raise RuntimeError(f'Invalid rate for music playback: {rate}')
        self.rate = rate

    def pause(self):
        self._streaming = False
        self.channel.pause()

    def unpause(self):
        self._streaming = True
        self.channel.unpause()

    def queue_next(self):
        """
        Will be driven by the pygame event queue, in response to
        end events on self.channel.
        """
        if self._streaming:
            self.index = (self.index + 1) % self.n_chunks
            self.channel.queue(self.chunks[self.rate][self.index]) 

    def chop(self, aim_for_even_chunks):
        if os.path.exists(self.dirpath) and not self.overwrite:
            print('found wave chunk directory')
            return
        wave_chop.chop_into_samples(self.src_path, self.dirpath, n_segments=-1,
                                    seconds_per_cut=self.segment_dur, start_note=0,
                                    aim_for_even_chunks=aim_for_even_chunks)

    def make_chunks(self, stretch=2):
        paths = (i.path for i in os.scandir(self.dirpath))
        self.chunks = {1: [], 2: []}
        self.arrs = {1: [], 2: []}
        self.durs = []
        for path in sorted(paths):
            dur = get_chunk_len(path)
            snd = pygame.mixer.Sound(path)
            arr = pygame.sndarray.samples(snd)
            stretched_arr = interp(arr, stretch)
            snd2 = pygame.sndarray.make_sound(stretched_arr)
            self.chunks[1].append(snd)
            self.chunks[2].append(snd2)
            self.arrs[1].append(arr)
            self.arrs[2].append(stretched_arr)
            self.durs.append(dur)
            #wc = WaveChunk(path, arr, stretched, snd, snd2, dur)
            #self.chunks.append(wc)

        



