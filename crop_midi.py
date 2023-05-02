import os
from glob import glob
from argparse import ArgumentParser
from typing import Optional

from miditoolkit import KeySignature, MidiFile, Note, TempoChange, TimeSignature


def crop_midi(file: str, starting_bar: Optional[int] = None, num_bars: Optional[int] = None):
    midi_obj = MidiFile(file)
    if starting_bar is None and num_bars is None:
        return midi_obj

    starting_tick = starting_bar * midi_obj.ticks_per_beat * 4 if starting_bar is not None else 0
    ending_tick = starting_tick + num_bars * midi_obj.ticks_per_beat * 4 if num_bars is not None else midi_obj.max_tick

    last_tempo_change, last_time_sig_change, last_key_sig_change = None, None, None
    if starting_tick > 0:
        last_tempo_change = [tempo for tempo in midi_obj.tempo_changes if tempo.time < starting_tick]
        last_tempo_change = last_tempo_change[-1] if len(last_tempo_change) > 0 else None
        last_time_sig_change = [
            time_sig for time_sig in midi_obj.time_signature_changes if time_sig.time < starting_tick
        ]
        last_time_sig_change = last_time_sig_change[-1] if len(last_time_sig_change) > 0 else None
        last_key_sig_change = [key_sig for key_sig in midi_obj.key_signature_changes if key_sig.time < starting_tick]
        last_key_sig_change = last_key_sig_change[-1] if len(last_key_sig_change) > 0 else None

    for i, track in enumerate(midi_obj.instruments):
        midi_obj.instruments[i].notes = [
            Note(
                pitch=note.pitch,
                start=note.start - starting_tick,
                end=note.end - starting_tick,
                velocity=note.velocity,
            )
            for note in track.notes
            if starting_tick <= note.start < ending_tick
        ]

    midi_obj.tempo_changes = [
        TempoChange(tempo=tempo.tempo, time=tempo.time - starting_tick)
        for tempo in midi_obj.tempo_changes
        if starting_tick <= tempo.time < ending_tick
    ]
    midi_obj.time_signature_changes = [
        TimeSignature(
            numerator=time_sig.numerator, denominator=time_sig.denominator, time=time_sig.time - starting_tick
        )
        for time_sig in midi_obj.time_signature_changes
        if starting_tick <= time_sig.time < ending_tick
    ]
    midi_obj.key_signature_changes = [
        KeySignature(key_name=key_sig.key_name, time=key_sig.time - starting_tick)
        for key_sig in midi_obj.key_signature_changes
        if starting_tick <= key_sig.time < ending_tick
    ]

    if last_tempo_change is not None:
        midi_obj.tempo_changes.insert(0, TempoChange(tempo=last_tempo_change.tempo, time=0))
    if last_time_sig_change is not None:
        midi_obj.time_signature_changes.insert(
            0,
            TimeSignature(
                numerator=last_time_sig_change.numerator, denominator=last_time_sig_change.denominator, time=0
            ),
        )
    if last_key_sig_change is not None:
        midi_obj.key_signature_changes.insert(0, KeySignature(key_name=last_key_sig_change.key_name, time=0))

    # ignore lyrics and markers
    return midi_obj

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--starting_bar', type=int, default=None)
    parser.add_argument('--num_bars', type=int, default=None)
    args = parser.parse_args()

    files = glob(os.path.join(args.src_dir, "*.mid"))
    dest_paths = [os.path.join(args.dest_dir, os.path.basename(file)) for file in files]
    for file, dest_path in zip(files, dest_paths):
        midi_obj = crop_midi(file, args.starting_bar, args.num_bars)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        midi_obj.dump(dest_path)
        print(f"Saved {dest_path}")
