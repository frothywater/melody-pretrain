"""
Copied from: https://github.com/slSeanWU/MusDr
"""

import itertools
import math
import os
import random
from typing import Optional

import mido
import numpy as np
import pandas as pd
import scipy.stats
from scipy.io import loadmat

"""
Default event encodings (ones used by the Jazz Transformer).
You may override the defaults in function arguments to suit your own vocabulary.
"""
BAR_EV = 192  # the ID of ``Bar`` event
POS_EVS = range(193, 257)  # the IDs of ``Position`` events
CHORD_EVS = {  # the IDs of Chord-related events
    "Chord-Tone": range(322, 334),
    "Chord-Type": range(346, 393),
    "Chord-Slash": range(334, 346),
}
N_PITCH_CLS = 12  # {C, C#, ..., Bb, B}


def get_event_seq(piece_csv, seq_col_name="ENCODING"):
    """
    Extracts the event sequence from a piece of music (stored in .csv file).
    NOTE: You should modify this function if you use different formats.

    Parameters:
      piece_csv (str): path to the piece's .csv file.
      seq_col_name (str): name of the column containing event encodings.

    Returns:
      list: the event sequence of the piece.
    """
    df = pd.read_csv(piece_csv, encoding="utf-8")
    return df[seq_col_name].astype("int32").tolist()


def get_chord_sequence(ev_seq, chord_evs):
    """
    Extracts the chord sequence (in string representation) from the input piece.
    NOTE: This function is vocabulary-dependent,
          you should implement a new one if a different vocab is used.

    Parameters:
      ev_seq (list): a piece of music in event sequence representation.
      chord_evs (dict of lists): [key] type of chord-related event --> [value] encodings belonging to the type.

    Returns:
      list of lists: The chord sequence of the input piece, each element (a list) being the representation of a single chord.
    """
    # extract chord-related tokens
    ev_seq = [x for x in ev_seq if any(x in chord_evs[typ] for typ in chord_evs.keys())]

    # remove grammar errors in sequence (vocabulary-dependent)
    legal_seq = []
    cnt = 0
    for i, ev in enumerate(ev_seq):
        cnt += 1
        if ev in chord_evs["Chord-Slash"] and cnt == 3:
            cnt = 0
            legal_seq.extend(ev_seq[i - 2 : i + 1])

    ev_seq = legal_seq
    assert not len(ev_seq) % 3
    chords = []
    for i in range(0, len(ev_seq), 3):
        chords.append(ev_seq[i : i + 3])

    return chords


def compute_histogram_entropy(hist):
    """
    Computes the entropy (log base 2) of a normalised histogram.

    Parameters:
      hist (ndarray): input pitch (or duration) histogram, should be normalised.

    Returns:
      float: entropy (log base 2) of the histogram.
    """
    return scipy.stats.entropy(hist) / np.log(2)


def get_pitch_histogram(ev_seq, pitch_evs=range(128), verbose=False):
    """
    Computes the pitch-class histogram from an event sequence.

    Parameters:
      ev_seq (list): a piece of music in event sequence representation.
      pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
      verbose (bool): whether to print msg. when ev_seq has no notes.

    Returns:
      ndarray: the resulting pitch-class histogram.
    """
    ev_seq = [x for x in ev_seq if x in pitch_evs]

    if not len(ev_seq):
        if verbose:
            print("[Info] The sequence contains no notes.")
        return None

    # compress sequence to pitch classes & get normalised counts
    ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
    ev_hist = ev_seq.value_counts(normalize=True)

    # make the final histogram
    hist = np.zeros((N_PITCH_CLS,))
    for i in range(N_PITCH_CLS):
        if i in ev_hist.index:
            hist[i] = ev_hist.loc[i]

    return hist


def get_onset_xor_distance(seq_a, seq_b, bar_ev_id, pos_evs, pitch_evs=range(128)):
    """
    Computes the XOR distance of onset positions between a pair of bars.

    Parameters:
      seq_a, seq_b (list): event sequence of a bar of music.
        IMPORTANT: for this implementation, a ``Note-Position`` event must appear before the associated ``Note-On``.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
      pitch_evs (list): encoding IDs of ``Note-On`` events.

    Returns:
      float: 0~1, the XOR distance between the 2 bars' (seq_a, seq_b) binary vectors of onsets.
    """
    # sanity checks
    assert seq_a[0] == bar_ev_id and seq_b[0] == bar_ev_id
    assert seq_a.count(bar_ev_id) == 1 and seq_b.count(bar_ev_id) == 1

    # compute binary onset vectors
    n_pos = len(pos_evs)

    def make_onset_vec(seq):
        cur_pos = -1
        onset_vec = np.zeros((n_pos,))
        for ev in seq:
            if ev in pos_evs:
                cur_pos = ev - pos_evs[0]
            if ev in pitch_evs:
                onset_vec[cur_pos] = 1
        return onset_vec

    a_onsets, b_onsets = make_onset_vec(seq_a), make_onset_vec(seq_b)

    # compute XOR distance
    dist = np.sum(np.abs(a_onsets - b_onsets)) / n_pos
    return dist


def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id, verbose=False):
    """
    Returns the designated crop (bars) of the input piece.

    Parameter:
      ev_seq (list): a piece of music in event sequence representation.
      start_bar (int): the starting bar of the crop.
      end_bar (int): the ending bar (inclusive) of the crop.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      verbose (bool): whether to print messages when unexpected operations happen.

    Returns:
      list: a cropped segment of music consisting of (end_bar - start_bar + 1) bars.
    """
    if start_bar < 0 or end_bar < 0:
        raise ValueError("Invalid start_bar: {}, or end_bar: {}.".format(start_bar, end_bar))

    # get the indices of ``Bar`` events
    ev_seq = np.array(ev_seq)
    bar_markers = np.where(ev_seq == bar_ev_id)[0]

    if start_bar > len(bar_markers) - 1:
        raise ValueError("start_bar: {} beyond end of piece.".format(start_bar))

    if end_bar < len(bar_markers) - 1:
        cropped_seq = ev_seq[bar_markers[start_bar] : bar_markers[end_bar + 1]]
    else:
        if verbose:
            print(
                "[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.".format(
                    end_bar, len(bar_markers) - start_bar
                )
            )
        cropped_seq = ev_seq[bar_markers[start_bar] :]

    return cropped_seq.tolist()


def read_fitness_mat(fitness_mat_file):
    """
    Reads and returns (as an ndarray) a fitness scape plot as a center-duration matrix.

    Parameters:
      fitness_mat_file (str): path to the file containing fitness scape plot.
        Accepted formats: .mat (MATLAB data), .npy (ndarray)

    Returns:
      ndarray: the fitness scapeplot encoded as a center-duration matrix.
    """
    ext = os.path.splitext(fitness_mat_file)[-1].lower()

    if ext == ".npy":
        f_mat = np.load(fitness_mat_file)
    elif ext == ".mat":
        mat_dict = loadmat(fitness_mat_file)
        f_mat = mat_dict["fitness_info"][0, 0][0]
        f_mat[np.isnan(f_mat)] = 0.0
    else:
        raise ValueError("Unsupported fitness scape plot format: {}".format(ext))

    for slen in range(f_mat.shape[0]):
        f_mat[slen] = np.roll(f_mat[slen], slen // 2)

    return f_mat


def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id=BAR_EV, pitch_evs=range(128), verbose=False):
    """
    Computes the average pitch-class histogram entropy of a piece.
    (Metric ``H``)

    Parameters:
      piece_ev_seq (list): a piece of music in event sequence representation.
      window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
      verbose (bool): whether to print msg. when a crop contains no notes.

    Returns:
      float: the average n-bar pitch-class histogram entropy of the input piece.
    """
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    n_bars = piece_ev_seq.count(bar_ev_id)
    if window_size > n_bars:
        print(
            "[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.".format(
                window_size
            )
        )
        window_size = n_bars

    # compute entropy of all possible segments
    pitch_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)

        pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
        if pitch_hist is None:
            if verbose:
                print("[Info] No notes in this crop: {}~{} bars.".format(st_bar, st_bar + window_size - 1))
            continue

        pitch_ents.append(compute_histogram_entropy(pitch_hist))

    return np.mean(pitch_ents)


def compute_piece_groove_similarity(
    piece_ev_seq, bar_ev_id=BAR_EV, pos_evs=POS_EVS, pitch_evs=range(128), max_pairs=1000
):
    """
    Computes the average grooving pattern similarity between all pairs of bars of a piece.
    (Metric ``GS``)

    Parameters:
      piece_ev_seq (list): a piece of music in event sequence representation.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
      pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
      max_pairs (int): maximum #(pairs) considered, to save computation overhead.

    Returns:
      float: 0~1, the average grooving pattern similarity of the input piece.
    """
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    # get every single bar & compute indices of bar pairs
    n_bars = piece_ev_seq.count(bar_ev_id)
    bar_seqs = []
    for b in range(n_bars):
        bar_seqs.append(get_bars_crop(piece_ev_seq, b, b, bar_ev_id))
    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    # compute pairwise grooving similarities
    grv_sims = []
    for p in pairs:
        grv_sims.append(
            1.0 - get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
        )

    return np.mean(grv_sims)


def compute_piece_chord_progression_irregularity(piece_ev_seq, chord_evs=CHORD_EVS, ngram=3):
    """
    Computes the chord progression irregularity of a piece.
    (Metric ``CPI``)

    Parameters:
      piece_ev_seq (list): a piece of music in event sequence representation.
      chord_evs (dict of lists): [key] type of chord-related event --> [value] encodings belonging to the type.
      ngram (int): the n-gram in chord progression considered (e.g., bigram, trigram, 4-gram ...), defaults to trigram.

    Returns:
      float: 0~1, the chord progression irregularity of the input piece, measured on the n-gram specified.
    """
    chord_seq = get_chord_sequence(piece_ev_seq, chord_evs)
    if len(chord_seq) <= ngram:
        return 1.0

    num_ngrams = len(chord_seq) - ngram
    unique_set = set()
    for i in range(num_ngrams):
        str_repr = "_".join(["-".join(str(x)) for x in chord_seq[i : i + ngram]])
        if str_repr not in unique_set:
            unique_set.add(str_repr)

    return len(unique_set) / num_ngrams


def compute_structure_indicator(mat_file, low_bound_sec=0, upp_bound_sec=128, sample_rate=2):
    """
    Computes the structureness indicator SI(low_bound_sec, upp_bound_sec) from fitness scape plot (stored in a MATLAB .mat file).
    (Metric ``SI``)

    Parameters:
      mat_file (str): path to the .mat file containing fitness scape plot of a piece. (computed by ``run_matlab_scapeplot.py``).
      low_bound_sec (int, >0): the smallest timescale (in seconds) you are interested to examine.
      upp_bound_sec (int, >0): the largest timescale (in seconds) you are interested to examine.
      sample_rate (int): sample rate (in Hz) of the input fitness scape plot.

    Returns:
      float: 0~1, the structureness indicator (i.e., max fitness value) of the piece within the given range of timescales.
    """
    assert (
        low_bound_sec > 0 and upp_bound_sec > 0
    ), "`low_bound_sec` and `upp_bound_sec` should be positive, got: low_bound_sec={}, upp_bound_sec={}.".format(
        low_bound_sec, upp_bound_sec
    )
    low_bound_ts = int(low_bound_sec * sample_rate) - 1
    upp_bound_ts = int(upp_bound_sec * sample_rate)
    f_mat = read_fitness_mat(mat_file)

    if low_bound_ts >= f_mat.shape[0]:
        score = 0
    else:
        score = np.max(f_mat[low_bound_ts:upp_bound_ts])

    return score


def convert_midi_to_events(midi_file: str, starting_bar: Optional[int] = None, num_bars: Optional[int] = None):
    beat_division = 16
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
            events.append(BAR_EV)
        elif notes[i - 1][0] != bar:
            bar_diff = bar - notes[i - 1][0]
            events += [BAR_EV] * bar_diff
        events.append(POS_EVS.start + position)
        events.append(pitch)
    return events
