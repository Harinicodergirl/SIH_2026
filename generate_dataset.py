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

# Required SNR levels
SNR_LEVELS = [15, 5, 0, -5, -15]

# Probability of using multiple noises
MULTI_NOISE_PROBABILITY = 0.5

# Number of noises in multiple-noise mixture
MIN_NOISES = 2
MAX_NOISES = 4

# Expected clean speech dataset
EXPECTED_SPEAKERS = 41
EXPECTED_RECORDINGS_PER_SPEAKER = 15

# Supported formats
AUDIO_EXTENSIONS = {".wav", ".flac"}

# Total target dataset
TOTAL_SAMPLES = 50000

# Reproducibility
GLOBAL_SEED = 2026


# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def load_audio(filepath, target_sr=TARGET_SR):
    """
    Load an audio file.

    Converts:
        stereo -> mono
        original sampling rate -> 16 kHz
    """

    audio, sr = sf.read(filepath, dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Remove NaN / Inf
    audio = np.nan_to_num(audio)

    # Resample
    if sr != target_sr:

        gcd = np.gcd(sr, target_sr)

        up = target_sr // gcd
        down = sr // gcd

        audio = resample_poly(
            audio,
            up,
            down
        )

    # Remove DC offset
    audio = audio - np.mean(audio)

    return audio.astype(np.float32)


def rms(signal):
    """
    Calculate RMS amplitude.
    """

    return np.sqrt(
        np.mean(signal ** 2) + 1e-12
    )


def normalize_rms(signal):
    """
    Normalize signal to RMS = 1.
    """

    value = rms(signal)

    if value < 1e-10:
        return signal

    return signal / value


def match_length(signal, target_length, rng):
    """
    Make noise exactly the same length as speech.

    If noise is shorter:
        repeat it.

    If noise is longer:
        randomly crop it.
    """

    if len(signal) == target_length:
        return signal

    if len(signal) < target_length:

        repeats = int(
            np.ceil(
                target_length / len(signal)
            )
        )

        signal = np.tile(
            signal,
            repeats
        )

        max_start = len(signal) - target_length

        if max_start > 0:
            start = rng.randint(
                0,
                max_start
            )

            signal = signal[
                start:start + target_length
            ]

        else:
            signal = signal[:target_length]

    else:

        max_start = len(signal) - target_length

        start = rng.randint(
            0,
            max_start
        )

        signal = signal[
            start:start + target_length
        ]

    return signal.astype(np.float32)


# ============================================================
# SPEAKER DATASET DISCOVERY
# ============================================================

def find_speech_dataset(speech_root):
    """
    Find:

        speaker(1)
        speaker(2)
        ...
        speaker(41)

    Each speaker should contain 15 recordings.
    """

    speech_root = Path(speech_root)

    speakers = {}

    for i in range(
        1,
        EXPECTED_SPEAKERS + 1
    ):

        speaker_name = f"speaker({i})"

        speaker_dir = (
            speech_root / speaker_name
        )

        if not speaker_dir.exists():

            raise RuntimeError(
                f"Missing speaker folder:\n"
                f"{speaker_dir}"
            )

        recordings = sorted([
            p
            for p in speaker_dir.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in AUDIO_EXTENSIONS
            )
        ])

        if len(recordings) != EXPECTED_RECORDINGS_PER_SPEAKER:

            raise RuntimeError(
                f"{speaker_name} contains "
                f"{len(recordings)} recordings.\n"
                f"Expected "
                f"{EXPECTED_RECORDINGS_PER_SPEAKER}."
            )

        speakers[speaker_name] = recordings

    return speakers


# ============================================================
# NOISE DATASET DISCOVERY
# ============================================================

def find_noise_dataset(
    dataset_root,
    speech_root
):
    """
    Find all noise files under dataset/.

    The speech directory is excluded.

    Example noise structure:

        dataset/
        ├── engine/
        ├── gunshot/
        ├── helicopter/
        ├── rain/
        ├── siren/
        ├── vehicle/
        ├── artillery/
        ├── drone/
        └── ...

    Returns:

        {
            "engine": [...],
            "gunshot": [...],
            ...
        }
    """

    dataset_root = Path(
        dataset_root
    ).resolve()

    speech_root = Path(
        speech_root
    ).resolve()

    noise_dataset = {}

    for folder in dataset_root.iterdir():

        if not folder.is_dir():
            continue

        # Never treat speech as noise
        if folder.resolve() == speech_root.resolve():
            continue

        files = [
            p
            for p in folder.rglob("*")
            if (
                p.is_file()
                and
                p.suffix.lower()
                in AUDIO_EXTENSIONS
            )
        ]

        if files:

            noise_dataset[
                folder.name
            ] = sorted(files)

    return noise_dataset


# ============================================================
# SELECT SPEAKER + RECORDING
# ============================================================

def select_speech(
    speakers,
    global_id
):
    """
    Select speakers in a balanced deterministic way.

    Example:

        sample 0  -> speaker(1)
        sample 1  -> speaker(2)
        ...
        sample 40 -> speaker(41)
        sample 41 -> speaker(1)

    This ensures that the 50,000 samples are
    approximately balanced across all 41 speakers.
    """

    speaker_index = (
        global_id % EXPECTED_SPEAKERS
    )

    speaker_name = (
        f"speaker({speaker_index + 1})"
    )

    recordings = speakers[
        speaker_name
    ]

    # Deterministic recording selection
    rng = random.Random(
        GLOBAL_SEED + global_id
    )

    recording = rng.choice(
        recordings
    )

    return (
        speaker_name,
        recording
    )


# ============================================================
# CREATE NOISE
# ============================================================

def create_noise(
    selected_noise_files,
    target_length,
    rng
):
    """
    Load and combine one or multiple noises.
    """

    components = []

    for noise_file in selected_noise_files:

        noise = load_audio(
            noise_file
        )

        if len(noise) == 0:
            continue

        noise = match_length(
            noise,
            target_length,
            rng
        )

        noise = normalize_rms(
            noise
        )

        components.append(noise)

    if not components:
        return None

    # Combine noises
    combined_noise = np.zeros(
        target_length,
        dtype=np.float32
    )

    for noise in components:

        combined_noise += noise

    # Normalize TOTAL noise
    combined_noise = normalize_rms(
        combined_noise
    )

    return combined_noise


# ============================================================
# MIX AT SPECIFIC SNR
# ============================================================

def mix_at_snr(
    speech,
    noise,
    snr_db
):
    """
    Mix noise with speech at requested SNR.

    SNR(dB) =
        20 log10(
            RMS_speech / RMS_noise
        )

    Therefore:

        noise_RMS =
        speech_RMS / 10^(SNR/20)
    """

    speech_rms = rms(
        speech
    )

    noise_rms = rms(
        noise
    )

    if noise_rms < 1e-10:
        return speech

    desired_noise_rms = (
        speech_rms
        /
        (10 ** (snr_db / 20.0))
    )

    noise_scaled = (
        noise
        *
        (
            desired_noise_rms
            /
            noise_rms
        )
    )

    mixed = (
        speech
        +
        noise_scaled
    )

    return mixed.astype(
        np.float32
    )


# ============================================================
# CLIPPING PROTECTION
# ============================================================

def prevent_clipping(audio):
    """
    Scale mixture if necessary.

    This does NOT change SNR because
    both speech and noise are scaled together.
    """

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0.999:

        audio = (
            audio / peak
        ) * 0.999

    return audio.astype(
        np.float32
    )


# ============================================================
# GENERATE DATA
# ============================================================

def generate_samples(
    worker_name,
    start_id,
    end_id,
    speakers,
    noise_dataset,
    output_root
):

    # --------------------------------------------------------
    # Worker output folder
    # --------------------------------------------------------

    worker_dir = (
        Path(output_root)
        / worker_name
    )

    audio_dir = (
        worker_dir
        / "audio"
    )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = (
        worker_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # Noise categories
    # --------------------------------------------------------

    noise_categories = list(
        noise_dataset.keys()
    )

    if len(noise_categories) == 0:

        raise RuntimeError(
            "No noise categories found."
        )

    if (
        len(noise_categories)
        < MIN_NOISES
    ):

        raise RuntimeError(
            f"Need at least "
            f"{MIN_NOISES} noise categories "
            f"for multiple-noise mixing."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = []

    total = end_id - start_id

    print()
    print("=" * 70)
    print(f"WORKER       : {worker_name}")
    print(f"START ID     : {start_id}")
    print(f"END ID       : {end_id - 1}")
    print(f"SAMPLES      : {total}")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Generate each sample
    # --------------------------------------------------------

    for global_id in range(
        start_id,
        end_id
    ):

        # --------------------------------------------
        # Deterministic RNG
        # --------------------------------------------

        seed = (
            GLOBAL_SEED
            + global_id
        )

        rng = random.Random(
            seed
        )

        # --------------------------------------------
        # SNR
        # --------------------------------------------

        # Equal distribution:
        #
        # 15 dB
        # 5 dB
        # 0 dB
        # -5 dB
        # -15 dB

        snr_db = SNR_LEVELS[
            global_id % len(SNR_LEVELS)
        ]

        # --------------------------------------------
        # Select speaker
        # --------------------------------------------

        (
            speaker_name,
            speech_file
        ) = select_speech(
            speakers,
            global_id
        )

        speech = load_audio(
            speech_file
        )

        if len(speech) == 0:

            print(
                f"Skipping {global_id}: "
                f"empty speech file"
            )

            continue

        # --------------------------------------------
        # Select single/multiple noise
        # --------------------------------------------

        use_multiple_noise = (
            rng.random()
            < MULTI_NOISE_PROBABILITY
        )

        if use_multiple_noise:

            number_of_noises = rng.randint(
                MIN_NOISES,
                min(
                    MAX_NOISES,
                    len(noise_categories)
                )
            )

            selected_categories = (
                rng.sample(
                    noise_categories,
                    number_of_noises
                )
            )

            noise_files = []

            for category in selected_categories:

                file = rng.choice(
                    noise_dataset[
                        category
                    ]
                )

                noise_files.append(
                    file
                )

            noise_type = "multiple"

        else:

            selected_categories = [
                rng.choice(
                    noise_categories
                )
            ]

            category = (
                selected_categories[0]
            )

            noise_files = [
                rng.choice(
                    noise_dataset[
                        category
                    ]
                )
            ]

            noise_type = "single"

        # --------------------------------------------
        # Create noise
        # --------------------------------------------

        noise = create_noise(
            noise_files,
            len(speech),
            rng
        )

        if noise is None:

            print(
                f"Skipping {global_id}: "
                f"invalid noise"
            )

            continue

        # --------------------------------------------
        # Mix
        # --------------------------------------------

        mixed = mix_at_snr(
            speech,
            noise,
            snr_db
        )

        # --------------------------------------------
        # Prevent clipping
        # --------------------------------------------

        mixed = prevent_clipping(
            mixed
        )

        # --------------------------------------------
        # Filename
        # --------------------------------------------

        filename = (
            f"sample_{global_id:06d}"
            f"_speaker_{speaker_name}"
            f"_snr_{snr_db:+03d}dB"
            f"_{noise_type}.wav"
        )

        output_file = (
            audio_dir
            / filename
        )

        # --------------------------------------------
        # Save noisy audio
        # --------------------------------------------

        sf.write(
            output_file,
            mixed,
            TARGET_SR,
            subtype="PCM_16"
        )

        # --------------------------------------------
        # Metadata
        # --------------------------------------------

        metadata.append([

            global_id,

            filename,

            speaker_name,

            speech_file.name,

            str(speech_file),

            noise_type,

            "|".join(
                selected_categories
            ),

            "|".join(
                str(f)
                for f in noise_files
            ),

            snr_db,

            TARGET_SR,

            len(mixed) / TARGET_SR
        ])

        # --------------------------------------------
        # Progress
        # --------------------------------------------

        completed = (
            global_id
            - start_id
            + 1
        )

        if (
            completed % 100 == 0
            or
            completed == total
        ):

            percentage = (
                completed
                /
                total
                *
                100
            )

            print(
                f"[{worker_name}] "
                f"{completed}/{total} "
                f"({percentage:.1f}%)"
            )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "global_id",

            "filename",

            "speaker_id",

            "speech_filename",

            "speech_path",

            "noise_type",

            "noise_categories",

            "noise_files",

            "snr_db",

            "sample_rate",

            "duration_seconds"

        ])

        writer.writerows(
            metadata
        )

    print()
    print("=" * 70)
    print(f"{worker_name} FINISHED")
    print(f"Generated : {len(metadata)}")
    print(f"Audio     : {audio_dir}")
    print(f"Metadata  : {metadata_path}")
    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate 50,000 noisy speech "
            "samples from 41 speakers."
        )
    )

    parser.add_argument(
        "--worker",
        required=True,
        choices=[
            "harini",
            "personA",
            "personC"
        ]
    )

    args = parser.parse_args()

    # ========================================================
    # PATHS
    # ========================================================

    PROJECT_ROOT = (
        Path(__file__).resolve().parent
    )

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

    # ========================================================
    # FIND SPEAKERS
    # ========================================================

    print()
    print("Checking clean speech dataset...")
    print()

    speakers = find_speech_dataset(
        SPEECH_ROOT
    )

    print(
        f"Found {len(speakers)} speakers."
    )

    total_recordings = sum(
        len(files)
        for files in speakers.values()
    )

    print(
        f"Found {total_recordings} "
        f"clean recordings."
    )

    print()

    # ========================================================
    # FIND NOISES
    # ========================================================

    print(
        "Searching for noise files..."
    )

    noise_dataset = find_noise_dataset(
        DATASET_ROOT,
        SPEECH_ROOT
    )

    print()

    print(
        f"Found {len(noise_dataset)} "
        f"noise categories:"
    )

    for category, files in noise_dataset.items():

        print(
            f"  {category}: "
            f"{len(files)} files"
        )

    print()

    # ========================================================
    # WORKER SPLITS
    # ========================================================

    worker_ranges = {

        # 16,667
        "harini": (
            0,
            16667
        ),

        # 16,667
        "personA": (
            16667,
            33334
        ),

        # 16,666
        "personC": (
            33334,
            50000
        )
    }

    start_id, end_id = (
        worker_ranges[
            args.worker
        ]
    )

    # ========================================================
    # GENERATE
    # ========================================================

    generate_samples(

        worker_name=args.worker,

        start_id=start_id,

        end_id=end_id,

        speakers=speakers,

        noise_dataset=noise_dataset,

        output_root=OUTPUT_ROOT
    )