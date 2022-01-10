#!/usr/bin/env python3
import wave
import os
import sys

def chop_into_samples(wav_path, out_dir, n_segments=-1, seconds_per_cut=1,
                        start_note=48, aim_for_even_chunks=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        ans = input(f'{out_dir} already exists, continue? [y/n]')
        if ans.lower() not in ['y','yes']:
            sys.exit(0)
    f = wave.open(wav_path, 'rb')

    fr = f.getframerate()
    n_frames = int(fr*seconds_per_cut)#*f.getnchannels())

    hitch_hiker_frames = 0
    target_frame_size = n_frames
    total_frames = f.getnframes()
    chunks, rem_frames = divmod(total_frames, target_frame_size)
    if aim_for_even_chunks and rem_frames > 0:
        ## Get chunk sizes to have minimal deviation
        ## Amortize remainder frames over all chunks
        n_amortized = rem_frames // chunks
        if n_amortized == 0: # there are more chunks than remainder frames
            # tack an extra 'hitch_hiker' frame on the first ''rem_frames'' chunks
            actual_frame_size = target_frame_size 
            hitch_hiker_frames = rem_frames 
            rem_frames = 0
        else:
            actual_frame_size = target_frame_size + n_amortized
            chunks, rem_frames = divmod(total_frames, actual_frame_size) 
            print(f'Evened out chunk size from {target_frame_size} to {actual_frame_size}')
        n_segments = chunks
        n_frames = actual_frame_size
         
    params = list(f.getparams())
    params[3] = n_frames
    note_midi_val = start_note
    i = 0
    while True:
        hitch_hiker = 1 if i < hitch_hiker_frames else 0
        if i == n_segments:
            break
        try:
            if aim_for_even_chunks and i == (n_segments-1):
                n_frames += rem_frames
            segment = f.readframes(n_frames + hitch_hiker)
            print(len(segment))
            if len(segment) == 0:
                break
        except wave.Error:
            break

        outpath = os.path.join(out_dir, f'{note_midi_val:04d}.wav') 
        with wave.open(outpath, 'wb') as g:
            g.setparams(params)
            g.writeframes(segment)
        note_midi_val += 1 
        i += 1
    f.close()

def make_midi_file_chromatic(start_note=24, octaves=6, note_len=480*4,
                                velocity=64):
    track = mido.MidiTrack()
    for i in range(12*octaves):
        msg = mido.Message('note_on', note=start_note+i,
                            velocity=velocity, time=note_len)
        track.append(msg)
    mid = mido.MidiFile()
    mid.tracks.append(track)
    name = f'chromatic_{start_note}_to_{start_note+12*octaves}.mid' 
    mid.save(name)
    return mid

DEFAULT_START_NOTE = 24
DEFAULT_SEGMENTS = -1
DEFAULT_SEGMENT_LENGTH = 1

if __name__ == '__main__':
    program, *args = sys.argv
    if args[0].lower() in ['h', '-h', 'help', '--help']:
        print(f"USAGE: {__file__} source_wav_path out_dir_name [start_note]"
                f" [segment length] [n_segments]")
        print(f"\t The default start_note is {DEFAULT_START_NOTE}")
        print(f"\t The default seg_len is {DEFAULT_SEGMENT_LENGTH}")
        print(f"\t If no value is given for n_segments, will slice until end of file")
        sys.exit(0)
    source_wav = args[0]
    out_dir = args[1]
    n_segments = DEFAULT_SEGMENTS
    start_note = DEFAULT_START_NOTE
    seg_len = DEFAULT_SEGMENT_LENGTH
    if len(args) > 2:
        start_note = int(args[2])
    if len(args) > 3:
        seg_len = float(args[3])
    if len(args) > 4:
        n_segments = int(args[4])

    print(source_wav, out_dir, n_segments, start_note)
    chop_into_samples(source_wav, out_dir, n_segments=n_segments,
                                        seconds_per_cut=seg_len, start_note=start_note)
