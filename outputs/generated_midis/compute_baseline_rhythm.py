from mido import MidiFile
from pathlib import Path

def compute_rhythm_div(midi_path):
    midi = MidiFile(midi_path)
    durations = []
    for track in midi.tracks:
        time = 0
        for msg in track:
            if not msg.is_meta:
                time += msg.time
                if msg.type in ['note_on', 'note_off']:
                    durations.append(msg.time)
    if len(durations) == 0:
        return 0
    return len(set(durations)) / len(durations)

# Since the script is inside generated_midis, subfolders are relative
rand_dir = Path("baseline_random/")
rand_vals = [compute_rhythm_div(f) for f in rand_dir.glob("*.mid")]
rand_avg = sum(rand_vals)/len(rand_vals)
print("Random Generator Rhythm Diversity:", rand_avg)

markov_dir = Path("baseline_markov/")
markov_vals = [compute_rhythm_div(f) for f in markov_dir.glob("*.mid")]
markov_avg = sum(markov_vals)/len(markov_vals)
print("Markov Chain Rhythm Diversity:", markov_avg)