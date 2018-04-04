# -*- coding, utf-8 -*-

from collections import OrderedDict
import numpy as np
import tensorflow as tf

CHORD_DATA_START_TIME = 0
CHORD_DATA_START_LEN  = 1
CHORD_DATA_ROOT_FREQ  = 2
CHORD_DATA_TYPE       = 3

#patterns of Chord, root note is normalized to 0
#Notice that the digit means the position in Twelve_tone,not the absolute musical alphabet 
NATIVE_CHORD_INFO = [
            # 3 notes
            [0, 'maj', [0, 4, 7]],
            [1, 'min', [0, 3, 7]],
            [2, 'dim', [0, 3, 6]],
            [3, 'aug', [0, 4, 8]],
            [4, 'sus2', [0, 2, 7]],
            [5, 'sus4', [0, 5, 7]],
            # 4 notes
            [6, '6', [0, 4, 7, 9]],
            [7, '7', [0, 4, 7, 10]],
            [8, '7-5', [0, 4, 6, 10]],
            [9, '7+5', [0, 4, 8, 10]],
            [10, '7sus4', [0, 5, 7, 10]],
            [11, 'm7', [0, 3, 7, 10]],
            [12, 'm7-5', [0, 3, 6, 10]],
            [13, 'dim6', [0, 3, 6, 9]],
            [14, 'M7', [0, 4, 7, 11]],
            [15, 'M7+5', [0, 4, 8, 11]],
            [16, 'mM7', [0, 3, 7, 11]],
            [17, 'add9', [0, 4, 7, 14]],
            [18, '2', [0, 4, 7, 14]],
            [19, 'add11', [0, 4, 7, 17]],
            [20, '4', [0, 4, 7, 17]],
            # 5 notes
            [21, '6/9', [0, 4, 7, 9, 14]],
            [22, '9', [0, 4, 7, 10, 14]],
            [23, 'm9', [0, 3, 7, 10, 14]],
            [24, 'M9', [0, 4, 7, 11, 14]],
            [25, '9sus4', [0, 5, 7, 10, 14]],
            [26, '7-9', [0, 4, 7, 10, 13]],
            [27, '7+9', [0, 4, 7, 10, 15]],
            [28, '11', [0, 7, 10, 14, 17]],
            [29, '7+11', [0, 4, 7, 10, 18]],
            [30, '7-13', [0, 4, 7, 10, 20]],
            # 6 notes
            [31, '13', [0, 4, 7, 10, 14, 21]],
        ]

Twelve_tone = [
    [0,'C'],
    [1,'C#'],
    [2,'D'],
    [3,'D#'],
    [4,'E'],
    [5,'F'],
    [6,'F#'],
    [7,'G'],
    [8,'G#'],
    [9,'A'],
    [10,'A#'],
    [11,'B'],
]


class Chord_Tools:
    def __init__(self):

        self.MAX_CHORD_NUM = len(NATIVE_CHORD_INFO)
        
    
        #Given the root node and chord_index,index in one_hot table can be calculated as 
        #index = 12*chord_index + root_note
        self.chord_one_hot_table = np.eye(self.MAX_CHORD_NUM) 

    #Given a group of notes that may compose a chord, find the possible chord in NATIVE_CHORD_INFO table
    def get_chord(self,notes):
        for chord_info in NATIVE_CHORD_INFO:
            if chord_info[2] == notes:
                return self.chord_one_hot_table[chord_info[0]]

        return None

    def get_chord_and_root(self,notes):
        record_notes = notes[:]

        #make a copy of the input notes
        for i in range(len(record_notes)):
            record_notes.insert(i+i+1,i)
        record_notes = np.reshape(record_notes,(-1,2))

        #find all permutations of the given notes
        import itertools
        record_notes = [ [n[0]%12,n[1]] for n in  record_notes ]
        all_permutations_notes = list(itertools.permutations(record_notes))

        #Check each permutation to find the possible chord and root note
        for permutations_note in all_permutations_notes:
            delta  = permutations_note[0][0]
            notes_to_be_checked = []
            root_index = permutations_note[0][1]
            for notes_info in permutations_note:
                note = notes_info[0] - delta 
                if note < 0:note += 12 
                notes_to_be_checked.append(note)
            chord = self.get_chord(notes_to_be_checked)
            if chord is not None:
                return notes[root_index],chord
        

        return None,None

    #Tool function to show the name of chord,like Cdim7
    def convert_to_chord_name(self,one_hot,root_note):
        type_index = np.argmax(one_hot)
        return Twelve_tone[int(root_note%12)][1]+NATIVE_CHORD_INFO[type_index][1],NATIVE_CHORD_INFO[type_index][2]
    
    def get_notes_from_batch_chord(self,data): 
        out_data_list = []
        for step in data:
            chord_notes = self.get_notes_from_chord(step[CHORD_DATA_ROOT_FREQ],step[CHORD_DATA_TYPE:])
            note_list = []
            for i in range(len(chord_notes)):
                note_list.append(step[CHORD_DATA_START_TIME])
                note_list.append(step[CHORD_DATA_START_LEN])
                note_list.append(chord_notes[i])
                out_data_list.append(note_list)
                note_list = []

        #out_data_tensor = tf.stack(out_data_list)
        #return out_data_tensor
        return out_data_list

    def get_notes_from_chord(self,root_note,one_hot):
        type_index = np.argmax(one_hot)
        chord_basic_notes = NATIVE_CHORD_INFO[type_index][2][:]
        chord_notes = [int(note + root_note) for note in chord_basic_notes]

        return chord_notes


def main():
    '''with tf.Session() as sess:
        chord_tools = Chord_Tools(sess,True)
        notes_pos = [24,28,31]
        root_note,notes_list = chord_tools.format_chord_note_and_get_root(notes_pos)
        chord = chord_tools.get_chord(root_note,notes_list)
        print(chord)'''

if __name__ == '__main__':
    main()