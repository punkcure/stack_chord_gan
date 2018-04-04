import pretty_midi as midi
import midi as py_midi
import numpy as np 
from Chord_utils import *
import os
import math
import tensorflow as tf


CHORD_DATA_START_TIME = 0
CHORD_DATA_START_LEN  = 1
CHORD_DATA_ROOT_FREQ  = 2
CHORD_DATA_TYPE       = 3

NOTE_START_FROM_PREV = 0
NOTE_START_LEN  = 1
NOTE_FREQ       = 2
NOTE_VELOCITY   = 3

DEBUG = True


def debug_print(str):
    if DEBUG:
        print(str)

class midi_utils:
    def __init__(self):
        self.chord_tool = Chord_Tools()
        self.batch_point = 0
        self.chord_info = []
        #todo
        self.output_ticks_per_quarter_note = 384.0

        #+1+1+1 means concat begin_time,length,and root_note,each one is a one-divesion vector
        self.FEATURE_SHAPE = self.chord_tool.MAX_CHORD_NUM+1+1+1

    def get_midi_pattern(self,song_data):
        #song_data:a tensor with shape[length,FEATURE_SHAPE] where FEATURE_SHAPE is [start_from_prev,length,freq,velocity]
        midi_pattern = py_midi.Pattern([],self.output_ticks_per_quarter_note)
        cur_track = py_midi.Track([])
        cur_track.append(py_midi.events.SetTempoEvent(tick=0,bpm=45))
        song_events_absolute_ticks = []
        abs_tick_note_beginning = 0.0

        for frame in song_data:
            abs_tick_note_beginning += frame[NOTE_START_FROM_PREV]
            
            tick_len = int(round(frame[NOTE_START_LEN]))
            #todo
            try:
                velocity = min(int(round(frame[NOTE_VELOCITY])),127)
            except: 
                velocity = 100
            tone = frame[NOTE_FREQ]
            if tone is None or velocity<0 or tick_len<=0: continue           

            song_events_absolute_ticks.append((abs_tick_note_beginning,
                                             py_midi.events.NoteOnEvent(
                                                   tick=0,
                                                   velocity=velocity,
                                                   pitch=tone)))
            song_events_absolute_ticks.append((abs_tick_note_beginning+tick_len,
                                             py_midi.events.NoteOffEvent(
                                                    tick=0,
                                                    velocity=0,
                                                    pitch=tone)))
            
        song_events_absolute_ticks.sort(key=lambda e: e[0])
        
        abs_tick_note_beginning = 0.0
        for abs_tick,event in song_events_absolute_ticks:
            rel_tick = abs_tick-abs_tick_note_beginning
            event.tick = int(round(rel_tick))
            cur_track.append(event)
            abs_tick_note_beginning=abs_tick
        
        cur_track.append(py_midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
        midi_pattern.append(cur_track)

        return midi_pattern
#********************************* save midi**********************************************    
    def save_midi_file(self,filename,song_data):
        midi_pattern = self.get_midi_pattern(song_data)
        py_midi.write_midifile(filename, midi_pattern)
    
    def save_batch_files(self,file_name_pre,data):
        #data is a tensor with shape[batch_size,step,FEATURE_SHAPE]
        batch_size = len(data)
        for i in range(batch_size):
            midi_data = data[i] #[step,FEATURE_SHAPE]   
            for j in xrange(len(midi_data)):
                d = self.freq_to_tone(midi_data[j][CHORD_DATA_ROOT_FREQ])
                midi_data[j][CHORD_DATA_ROOT_FREQ] = d['tone']
            midi_data = self.chord_tool.get_notes_from_batch_chord(midi_data)
            file_name = file_name_pre + '_'+str(i)+'.midi'
            self.save_midi_file(file_name,midi_data)
#******************************************************************************************

#*********************************read midi************************************************
#read all midi files in the path,process and record the data 
    def read_midi_files(self,path):
        file_names =  os.listdir(path)
        file_names = [path + file_name  for file_name in file_names]
        for file_name in file_names:
            chords_list = self.read_midi_file_and_process(file_name)
            if chords_list is not None:
                self.chord_info.append(chords_list)
        
        #todo shuffle the datas

#read one midi file
    def read_midi_file_and_process(self,file_name):
        #read a midi file ,split the chord track(convert to one-hot code) and notes track(convert to piano_roll form)
        midi_data = midi.PrettyMIDI(file_name)        
        debug_print('begin to read file:{}'.format(file_name))

        group_notes = []
        current_group_start_time = current_group_time_len = 0.0

        #check the validation of the midi file,find out the chord track
        chords_list = []
        chord_index = self.check_midi_data(midi_data)
        if chord_index is None:
            print('{} is a bad midi file without chord track!')
            return None
        chord_track_notes = midi_data.instruments[chord_index].notes[:]
        chord_track_notes.sort(key = lambda c: c.start)

        for note_data in chord_track_notes:
            if note_data.start > current_group_start_time and note_data.start > 0:              
                if len(group_notes) > 1:
                    # the order of appending operation is important ,shoud not be altered 
                    chord_data = []
                    root,chord_type = self.chord_tool.get_chord_and_root(group_notes)
                    chord_data.append(current_group_start_time)
                    chord_data.append(current_group_time_len)
                    chord_data.append(self.tone_to_freq(root))
                    chord_data.append(chord_type)
                    
                    if chord_type is not None:
                        chord_name,_ = self.chord_tool.convert_to_chord_name(chord_type,self.freq_to_tone(chord_data[CHORD_DATA_ROOT_FREQ])['tone'])
                        debug_print('Time: {} group notes : {} {} {} get chord {}'.format(current_group_start_time,group_notes[0],group_notes[1],group_notes[2],chord_name))
                        chords_list.append(chord_data)
                    else:
                        debug_print('******************Time: {} group notes : {} {} {}  get chord failed !'.format(current_group_start_time,group_notes[0],group_notes[1],group_notes[2]))  
                   
                    group_notes=[]
                    current_group_start_time = 0.0
                    current_group_time_len = 0.0
                
                group_notes.append(note_data.pitch)
                current_group_time_len = max(current_group_time_len,note_data.end - note_data.start)
                current_group_start_time = note_data.start
            else:
                group_notes.append(note_data.pitch)
                current_group_time_len = max(current_group_time_len,note_data.end - note_data.start)
                current_group_start_time = note_data.start
        
        # try:
        #    piano_roll = midi_data.get_piano_roll()
        #except:
        chords_list.sort(key = lambda c: c[0])
        return chords_list
#*********************************************************************************************************
    
    def check_midi_data(self,midi_data):
        debug_print('   check_midi_data:begin')
        if midi_data is not None:
            if len(midi_data.instruments) == 1:
                return None
            for i,track_native in enumerate(midi_data.instruments):
                track_notes = track_native.notes[:]
                debug_print('     check track:{}'.format(i))
                if len(track_notes) > 10:                    
                    possible_chord = 0
                    track_notes.sort(key = lambda c:c.start)
                    #random select 3 groups of  synchronous notes to check
                    for try_time in range(3):
                        begin_index = np.random.randint(len(track_notes))

                        baseline_time = track_notes[begin_index].start
                        match_count = 0
                        
                        debug_print('           trytime:{},begin_index:{},base_time:{}'.format(try_time,begin_index,baseline_time))

                        #search backward 1
                        if begin_index - 1 >= 0:
                            if track_notes[begin_index-1].start == baseline_time:
                                match_count += 1
                            debug_print('               -1 start={},match_count:{}'.format(track_notes[begin_index-1].start,match_count))
                        #search backward 2
                        if begin_index - 2 >= 0:
                            if track_notes[begin_index-2].start == baseline_time:
                                match_count += 1
                            debug_print('               -2 start={},match_count:{}'.format(track_notes[begin_index-2].start,match_count))
                        #search forward 1
                        if begin_index + 1 < len(track_notes):
                            if track_notes[begin_index+1].start == baseline_time:
                                match_count += 1
                            debug_print('               +1 start={},match_count:{}'.format(track_notes[begin_index+1].start,match_count))
                        #search forward 2
                        if begin_index + 2 < len(track_notes):
                            if track_notes[begin_index+2].start == baseline_time:
                                match_count += 1
                            debug_print('               +2 start={},match_count:{}'.format(track_notes[begin_index+2].start,match_count))
                        if match_count >= 2:
                            possible_chord += 1
                        else:
                            continue
                    debug_print('              possible_chord:{}'.format(possible_chord))
                    if possible_chord >= 3:
                        return i
                
        return None

    def get_chord_info_len(self):
        return len(self.chord_info)
                        
    def get_batch_chord(self,batch_size,length,reset_point = False):
        if reset_point:
            self.batch_point = 0

        batch_chord_return = np.ndarray(shape = [batch_size,length,self.FEATURE_SHAPE])

        temp_chord_info = self.chord_info[self.batch_point : self.batch_point + batch_size]
        self.batch_point += batch_size

        for idx_chord_info in range(len(temp_chord_info)):
            random_start = 0
            if len(temp_chord_info[idx_chord_info]) > length:
                #randomly choose a piece of data which length is length
                random_start = np.random.randint(0,len(temp_chord_info[idx_chord_info])-length)
            
            per_length_chord_info = np.zeros(shape = [length,self.FEATURE_SHAPE])

            for idx_row in range(length)    :
                if random_start < len(temp_chord_info[idx_chord_info]):
                    time_from_previous = 0.0
                    if random_start > 0:
                        time_from_previous = temp_chord_info[idx_chord_info][random_start][CHORD_DATA_START_TIME] - \
                            temp_chord_info[idx_chord_info][random_start-1][CHORD_DATA_START_TIME]

                    per_length_chord_info[idx_row,CHORD_DATA_START_TIME] = time_from_previous
                    per_length_chord_info[idx_row,CHORD_DATA_START_LEN] = temp_chord_info[idx_chord_info][random_start][CHORD_DATA_START_LEN]
                    per_length_chord_info[idx_row,CHORD_DATA_ROOT_FREQ] = temp_chord_info[idx_chord_info][random_start][CHORD_DATA_ROOT_FREQ]
                    per_length_chord_info[idx_row,CHORD_DATA_TYPE:] = temp_chord_info[idx_chord_info][random_start][CHORD_DATA_TYPE]
                random_start += 1

            batch_chord_return[idx_chord_info,:] = per_length_chord_info

            
        return batch_chord_return

    def tone_to_freq(self,tone):
        """
        returns the frequency of a tone. 

        formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
        """
        return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0

    def freq_to_tone(self,freq):
        """
        returns a dict d where
        d['tone'] is the base tone in midi standard
        d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.

        formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
        """
        if freq <= 0.0:
            return None
        float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
        int_tone = int(float_tone)
        cents = int(1200*math.log(float(freq)/self.tone_to_freq(int_tone), 2))
        while int_tone < 0:   int_tone += 12
        while int_tone > 127: int_tone -= 12
   
        return {'tone': int_tone, 'cents': cents}

def main():
    midi_tools = midi_utils()
    #chords_list = midi_tools.read_midi_file_and_process('/home/punkcure/TrainningResource/Nottingham/train/ashover_simple_chords_15.mid')
    midi_tools.read_midi_files('E:\\MLCode\\TrainningSource\\Nottingham\\Nottingham\\train\\')

    for epoch in range(500):
        batch_chords = midi_tools.get_batch_chord(batch_size=50,length=30)
        pass
        pass
        pass


if __name__ == '__main__':
    main()
