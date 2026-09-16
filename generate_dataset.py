import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SR = 16000

SNR_LEVELS = [15, 5, 0, -5, -15]

# Approximately 50% single-noise and 50% multiple-noise
MULTI_NOISE_PROBABILITY = 0.5

# Number of noises when using multiple-noise mode
MIN_NOISES = 2
MAX_NOISES = 4

# Supported audio files
AUDIO_EXTENSIONS = {".wav", ".flac"}


# ============================================================
# AUDIO UTILITIES
# ============================================================

def load_audio(filepath, target_sr=TARGET_SR):
    """
    Load audio as mono float32 and resample to target_sr.
    """

    audio, sr = sf.read(filepath, dtype="float32")

    # Convert stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Remove NaN / Inf
    audio = np.nan_to_num(audio)

    # Resample if required
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)

        up = target_sr // gcd
        down = sr // gcd

        audio = resample_poly(audio, up, down)

    return audio.astype(np.float32)


def rms(signal):
    """
    Calculate RMS energy.
    """

    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def normalize_rms(signal):
    """
    Normalize signal to RMS = 1.
    """

    r = rms(signal)

    if r < 1e-10:
        return signal

    return signal / r


def match_length(signal, target_length, rng):
    """
    Make signal exactly target_length.

    If shorter:
        Repeat it.

    If longer:
        Randomly crop it.
    """

    if len(signal) == target_length:
        return signal

    if len(signal) < target_length:

        # Repeat until long enough
        repeats = int(np.ceil(target_length / len(signal)))

        signal = np.tile(signal, repeats)

        # Random starting point
        max_start = len(signal) - target_length

        if max_start > 0:
            start = rng.randint(0, max_start)
            signal = signal[start:start + target_length]
        else:
            signal = signal[:target_length]

    else:

        # Random crop
        max_start = len(signal) - target_length

        start = rng.randint(0, max_start)

        signal = signal[start:start + target_length]

    return signal.astype(np.float32)


def choose_speech_segment(speech_file, rng):
    """
    Load speech and return the complete speech signal.

    Later, this can be changed to fixed-duration chunks.
    """

    speech = load_audio(speech_file)

    # Remove very short files
    if len(speech) < TARGET_SR:
        return None

    # Remove DC offset
    speech = speech - np.mean(speech)

    return speech


# ============================================================
# NOISE MIXING
# ============================================================

def create_noise(noise_files, target_length, rng):
    """
    Create a noise signal of target_length.

    noise_files:
        list containing either one noise file
        or multiple noise files.
    """

    noise_components = []

    for noise_file in noise_files:

        noise = load_audio(noise_file)

        if len(noise) == 0:
            continue

        noise = noise - np.mean(noise)

        # Make same length as speech
        noise = match_length(
            noise,
            target_length,
            rng
        )

        # Normalize each noise before combining
        noise = normalize_rms(noise)

        noise_components.append(noise)

    if len(noise_components) == 0:
        return None

    # Combine multiple noises
    noise = np.zeros(target_length, dtype=np.float32)

    for component in noise_components:
        noise += component

    # Normalize the FINAL combined noise.
    #
    # This is important because we want the TOTAL noise
    # to have the requested SNR.
    noise = normalize_rms(noise)

    return noise


def mix_at_snr(speech, noise, snr_db):
    """
    Mix speech and noise at a specified SNR.

    SNR = 10 log10(Pspeech / Pnoise)

    Therefore:

        noise_power = speech_power / 10^(SNR/10)
    """

    speech_rms = rms(speech)
    noise_rms = rms(noise)

    if noise_rms < 1e-10:
        return speech

    # Desired noise RMS
    desired_noise_rms = speech_rms / (10 ** (snr_db / 20.0))

    # Scale noise
    noise_scaled = noise * (
        desired_noise_rms / noise_rms
    )

    # Mix
    mixed = speech + noise_scaled

    return mixed.astype(np.float32)


def prevent_clipping(audio):
    """
    Prevent digital clipping.

    Scaling the complete mixture does NOT change its SNR.
    """

    peak = np.max(np.abs(audio))

    if peak > 0.999:

        audio = audio / peak * 0.999

    return audio.astype(np.float32)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_audio_files(root):
    """
    Recursively find WAV/FLAC files.
    """

    files = []

    root = Path(root)

    for path in root.rglob("*"):

        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)

    return sorted(files)


def find_speech_files(speech_root):
    """
    Find all clean speech files.

    Expected:

        raw/signals/speech/
            speaker001/
            speaker002/
            ...
    """

    files = find_audio_files(speech_root)

    return files


def find_noise_files(dataset_root, speech_root):
    """
    Find noise files while excluding the speech directory.

    This allows:

        dataset/
            engine/
            gunshot/
            helicopter/
            rain/
            siren/
            ...

    """

    dataset_root = Path(dataset_root).resolve()
    speech_root = Path(speech_root).resolve()

    noise_files = []

    for path in dataset_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Never treat clean speech as noise
        try:
            path.relative_to(speech_root)
            continue
        except ValueError:
            pass

        noise_files.append(path)

    return sorted(noise_files)


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_samples(
    worker_name,
    start_id,
    end_id,
    speech_files,
    noise_files,
    output_root,
    seed=2026
):

    output_root = Path(output_root)

    worker_output = output_root / worker_name
    audio_output = worker_output / "audio"

    audio_output.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = worker_output / "metadata.csv"

    # Metadata
    metadata_rows = []

    print()
    print("=" * 60)
    print(f"WORKER: {worker_name}")
    print(f"SAMPLE RANGE: {start_id} -> {end_id - 1}")
    print(f"NUMBER OF SAMPLES: {end_id - start_id}")
    print("=" * 60)

    for global_id in range(start_id, end_id):

        # ----------------------------------------------------
        # Deterministic RNG
        # ----------------------------------------------------
        #
        # Every sample gets its own seed.
        #
        # This means sample 12345 will always be generated
        # using the same random choices.
        #
        sample_rng = random.Random(
            seed + global_id
        )

        np_rng = np.random.default_rng(
            seed + global_id
        )

        # ----------------------------------------------------
        # Select SNR
        # ----------------------------------------------------
        #
        # Gives an approximately/evenly distributed set.
        #
        snr_db = SNR_LEVELS[
            global_id % len(SNR_LEVELS)
        ]

        # ----------------------------------------------------
        # Select speaker
        # ----------------------------------------------------

        speech_file = sample_rng.choice(
            speech_files
        )

        speech = choose_speech_segment(
            speech_file,
            sample_rng
        )

        if speech is None:
            print(
                f"Skipping {global_id}: "
                f"speech too short"
            )
            continue

        # ----------------------------------------------------
        # Decide single vs multiple noise
        # ----------------------------------------------------

        use_multiple_noise = (
            sample_rng.random()
            < MULTI_NOISE_PROBABILITY
        )

        if use_multiple_noise:

            number_of_noises = sample_rng.randint(
                MIN_NOISES,
                MAX_NOISES
            )

            # Cannot select more unique noises than available
            number_of_noises = min(
                number_of_noises,
                len(noise_files)
            )

            selected_noise_files = sample_rng.sample(
                noise_files,
                number_of_noises
            )

            noise_type = "multiple"

        else:

            selected_noise_files = [
                sample_rng.choice(noise_files)
            ]

            noise_type = "single"

        # ----------------------------------------------------
        # Create noise
        # ----------------------------------------------------

        noise = create_noise(
            selected_noise_files,
            len(speech),
            sample_rng
        )

        if noise is None:
            print(
                f"Skipping {global_id}: "
                f"invalid noise"
            )
            continue

        # ----------------------------------------------------
        # Mix
        # ----------------------------------------------------

        mixed = mix_at_snr(
            speech,
            noise,
            snr_db
        )

        # ----------------------------------------------------
        # Prevent clipping
        # ----------------------------------------------------

        mixed = prevent_clipping(mixed)

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        filename = (
            f"sample_{global_id:06d}"
            f"_snr_{snr_db:+03d}dB"
            f"_{noise_type}.wav"
        )

        output_file = audio_output / filename

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        sf.write(
            output_file,
            mixed,
            TARGET_SR,
            subtype="PCM_16"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        noise_names = [
            str(path.relative_to(Path(dataset_root)))
            if False
            else str(path)
            for path in selected_noise_files
        ]

        speaker_id = speech_file.parent.name

        metadata_rows.append([
            global_id,
            filename,
            speaker_id,
            str(speech_file),
            "|".join(noise_names),
            noise_type,
            snr_db,
            TARGET_SR,
            len(mixed) / TARGET_SR
        ])

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (global_id - start_id + 1) % 100 == 0
            or global_id == end_id - 1
        ):

            completed = global_id - start_id + 1
            total = end_id - start_id

            print(
                f"[{worker_name}] "
                f"{completed}/{total} "
                f"generated"
            )

    # ========================================================
    # Write metadata
    # ========================================================

    with open(
        metadata_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "global_id",
            "filename",
            "speaker_id",
            "speech_file",
            "noise_files",
            "noise_type",
            "snr_db",
            "sample_rate",
            "duration_seconds"
        ])

        writer.writerows(metadata_rows)

    print()
    print(
        f"Finished {worker_name}: "
        f"{len(metadata_rows)} samples"
    )
    print(
        f"Audio: {audio_output}"
    )
    print(
        f"Metadata: {metadata_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate noisy speech dataset"
    )

    parser.add_argument(
        "--worker",
        required=True,
        choices=[
            "harini",
            "personA",
            "personC"
        ],
        help="Which person's 1/3 of dataset to generate"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent

    DATASET_ROOT = (
        PROJECT_ROOT / "dataset"
    )

    SPEECH_ROOT = (
        DATASET_ROOT
        / "raw"
        / "signals"
        / "speech"
    )

    OUTPUT_ROOT = (
        DATASET_ROOT
        / "generated"
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    print("Searching for speech files...")

    speech_files = find_speech_files(
        SPEECH_ROOT
    )

    print(
        f"Found {len(speech_files)} "
        f"clean speech files."
    )

    print("Searching for noise files...")

    noise_files = find_noise_files(
        DATASET_ROOT,
        SPEECH_ROOT
    )

    print(
        f"Found {len(noise_files)} "
        f"noise files."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(speech_files) == 0:

        raise RuntimeError(
            f"No speech files found in:\n"
            f"{SPEECH_ROOT}"
        )

    if len(noise_files) == 0:

        raise RuntimeError(
            f"No noise files found in:\n"
            f"{DATASET_ROOT}"
        )

    if len(noise_files) < MIN_NOISES:

        raise RuntimeError(
            f"Need at least "
            f"{MIN_NOISES} noise files "
            f"for multiple-noise mixing."
        )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    TOTAL_SAMPLES = 50000

    worker_ranges = {

        "harini": (
            0,
            16667
        ),

        "personA": (
            16667,
            33334
        ),

        "personC": (
            33334,
            50000
        )
    }

    start_id, end_id = worker_ranges[
        args.worker
    ]

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generate_samples(
        worker_name=args.worker,
        start_id=start_id,
        end_id=end_id,
        speech_files=speech_files,
        noise_files=noise_files,
        output_root=OUTPUT_ROOT,
        seed=2026
    )