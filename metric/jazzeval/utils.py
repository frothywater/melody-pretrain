from typing import Optional
import mido
import math

beat_division = 16
units_per_bar = beat_division * 4

note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
chord_qualities = ["M", "m", "o", "+", "sus", "MM7", "Mm7", "mM7", "mm7", "o7", "%7", "+7", "+M7"]


def make_range(start: int, count: int):
    return start, range(start, start + count)


bar_event = 192
pos_start, pos_events = make_range(200, 64)
pitch_start, pitch_events = make_range(0, 128)
chord_tone_start, chord_tone_events = make_range(300, 12)
chord_type_start, chord_type_events = make_range(320, 13)
chord_slash_event = 340
chord_events = {
    "Chord-Tone": chord_tone_events,
    "Chord-Type": chord_type_events,
    "Chord-Slash": range(chord_slash_event, chord_slash_event + 1),
}


def convert_midi_to_events(midi_file: str, starting_bar: Optional[int] = None, num_bars: Optional[int] = None):
    midi = mido.MidiFile(midi_file)

    notes = []
    current_tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        if msg.time > 0:
            current_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            bar = current_tick // (midi.ticks_per_beat * 4)
            tick_in_bar = current_tick % (midi.ticks_per_beat * 4)
            position = round(tick_in_bar / midi.ticks_per_beat * beat_division)
            notes.append((bar, position, msg.note))

    starting_bar = starting_bar if starting_bar is not None else 0
    ending_bar = starting_bar + num_bars if num_bars is not None else math.inf
    notes = [note for note in notes if starting_bar <= note[0] < ending_bar]

    events = []
    for i, (bar, position, pitch) in enumerate(notes):
        if i == 0:
            events.append(bar_event)
        elif notes[i - 1][0] != bar:
            bar_diff = bar - notes[i - 1][0]
            events += [bar_event] * bar_diff
        events.append(pos_start + position)
        events.append(pitch_start + pitch)
    return events


if __name__ == "__main__":
    events = convert_midi_to_events("data/1.mid")
