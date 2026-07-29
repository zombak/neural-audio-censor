import os
import sys
import re
import glob
import time
import threading
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional, Dict, Tuple, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from pydub import AudioSegment
from pydub import generators as pydub_gen

try:
    import faster_whisper
except ImportError:
    faster_whisper = None

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

import logging
from dotenv import load_dotenv


# --- Constants ---
APP_TITLE = "Neural Audio Censor by Zombak (Zavtracast Podcast)"

UI_LANG_MAP = {
    "russian": "Russian",
    "english": "English",
    "german": "German",
    "french": "French",
    "spanish": "Spanish",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "ukrainian": "Ukrainian",
    "polish": "Polish"
}

CENSOR_LANG_CODE_MAP = {
    "Russian": "ru",
    "English": "en"
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s",
    datefmt="%H:%M:%S"
)

load_dotenv()

# =============================
# External Config Loader
# =============================
class ConfigLoader:
    @staticmethod
    def load_ui_strings(lang: str) -> Dict[str, str]:
        """Loads UI translation strings from a text file."""
        filename = f"ui-{lang}.txt"
        strings = {}
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        strings[k] = v.replace("\\n", "\n")
            return strings
        except FileNotFoundError:
            logging.error(f"UI file {filename} not found!")
            return {}

    @staticmethod
    def load_mat_regex(lang_name: str) -> re.Pattern:
        """Loads the profanity regex pattern based on the selected language."""
        lang_code = CENSOR_LANG_CODE_MAP.get(lang_name, "en")
        filename = f"words-{'russian' if lang_code == 'ru' else 'english'}.txt"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                pattern_str = f.read().strip()
                return re.compile(pattern_str, re.IGNORECASE)
        except FileNotFoundError:
            logging.error(f"Regex file {filename} not found! Returning empty pattern.")
            return re.compile(r'(?!.)', re.IGNORECASE)

# =============================
# Neural Network Handlers
# =============================
LANGUAGE_OPTIONS = {
    "Whisper": [
        ("Russian", "ru"), ("English", "en"), ("German", "de"), 
        ("French", "fr"), ("Spanish", "es"), ("Chinese", "zh"), 
        ("Japanese", "ja"), ("Korean", "ko"), ("Auto", None)
    ],
    "GigaAM": [
        ("Russian", "ru"), ("English", "en")
    ],
    "Parakeet": [
        ("Russian", "ru"), ("English", "en")
    ]


}

WHISPER_VERSION_MAP = {
    "tiny": "WHISPER_TINY_PATH",
    "base": "WHISPER_BASE_PATH",
    "small": "WHISPER_SMALL_PATH",
    "medium": "WHISPER_MEDIUM_PATH",
    "large-v3": "WHISPER_LARGE_V3_PATH"
}

class BaseTranscriber:
    """Abstract base class for transcription engines."""
    def __init__(self, model_path_or_name: str):
        self.model_source = model_path_or_name
        self.model = None

    def load_model(self):
        raise NotImplementedError

    def transcribe(self, audio_path: str, language: str = "ru"):
        raise NotImplementedError

class WhisperTranscriber(BaseTranscriber):
    """Implementation of Whisper using faster-whisper."""
    def load_model(self):
        if faster_whisper is None:
            raise ImportError("Library 'faster-whisper' is not installed")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        is_local = os.path.exists(self.model_source) and os.path.isdir(self.model_source)
        
        self.model = faster_whisper.WhisperModel(
            self.model_source, 
            device=device, 
            compute_type=compute_type,
            local_files_only=is_local
        )

    def transcribe(self, audio_path: str, language: str = "ru"):
        segments, info = self.model.transcribe(
            audio_path, word_timestamps=True, condition_on_previous_text=False, 
            vad_filter=True, language=language
        )
        return segments

class GigaAMTranscriber(BaseTranscriber):
    """Implementation of GigaAM using Hugging Face Transformers."""
    def load_model(self):
        if AutoModel is None:
            raise ImportError("Library 'transformers' is not installed")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = os.path.exists(self.model_source)
        
        self.model = AutoModel.from_pretrained(
            self.model_source, 
            revision="large_ctc", 
            trust_remote_code=True, 
            local_files_only=local_only
        ).to(device)

    def transcribe(self, audio_path: str, language: str = "ru"):
        return self.model.transcribe(audio_path)

# Import NeMo with minimal suppression (logs will still appear but app works)
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None


class ParakeetTranscriber(BaseTranscriber):
    """Implementation of Parakeet TDT using NeMo toolkit."""
    CHUNK_DURATION_MS = 60000

    def load_model(self):
        if nemo_asr is None:
            raise ImportError("Library 'nemo_toolkit[asr]' is not installed")

        model_path = self.model_source

        # If local folder, find .nemo checkpoint inside it
        if os.path.isdir(model_path):
            nemo_files = [f for f in os.listdir(model_path) if f.endswith(".nemo")]
            if nemo_files:
                model_path = os.path.join(model_path, nemo_files[0])

        # Use restore_from for local .nemo files, from_pretrained for HF names
        if model_path.endswith(".nemo") and os.path.exists(model_path):
            self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)

        # Enable local attention for long-form audio to prevent CUDA OOM on consumer GPUs
        try:
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        except Exception as e:
            logging.warning(f"Failed to enable local attention mode for Parakeet: {e}")

    def transcribe(self, audio_path: str, language: str = "ru"):
        # Parakeet requires 16kHz mono audio - convert if needed
        original_audio = AudioSegment.from_wav(audio_path)
        converted = original_audio.set_frame_rate(16000).set_channels(1)

        tmp_dir = tempfile.mkdtemp()
        chunk_paths: List[str] = []

        try:
            # Split audio into chunks to avoid CUDA OOM on long files
            duration_ms = len(converted)
            for start_ms in range(0, duration_ms, self.CHUNK_DURATION_MS):
                end_ms = min(start_ms + self.CHUNK_DURATION_MS, duration_ms)
                chunk_path = os.path.join(tmp_dir, f"chunk_{start_ms}.wav")
                converted[start_ms:end_ms].export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)

            # Transcribe each chunk separately and merge results
            all_word_timestamps: List[Dict] = []
            all_segment_timestamps: List[Dict] = []
            cumulative_offset_sec = 0.0

            for chunk_path in chunk_paths:
                output = self.model.transcribe([chunk_path], timestamps=True)
                chunk_result = output[0]

                # Get chunk duration for offset calculation
                chunk_audio = AudioSegment.from_wav(chunk_path)
                chunk_duration_sec = len(chunk_audio) / 1000.0

                # Merge word timestamps with adjusted offsets - try both .timestamp and .timestep
                ts_attr = getattr(chunk_result, 'timestamp', None) or getattr(chunk_result, 'timestep', None)
                if isinstance(ts_attr, dict):
                    words = ts_attr.get("word", [])
                    for w in words:
                        # Try multiple possible key names for the text content
                        token_val = w.get("token", "") or w.get("text", "") or w.get("word", "")
                        all_word_timestamps.append({
                            "start": w["start"] + cumulative_offset_sec,
                            "end": w["end"] + cumulative_offset_sec,
                            "token": token_val
                        })

                    # Merge segment timestamps with adjusted offsets
                    segments = ts_attr.get("segment", [])
                    for s in segments:
                        all_segment_timestamps.append({
                            "start": s["start"] + cumulative_offset_sec,
                            "end": s["end"] + cumulative_offset_sec,
                            "segment": s.get("segment", "")
                        })

                cumulative_offset_sec += chunk_duration_sec

            # Build merged result with same structure as single-file output
            class MergedResult:
                def __init__(self, words: List[Dict], segments: List[Dict]):
                    self.timestamp = {"word": words, "segment": segments}

            return MergedResult(all_word_timestamps, all_segment_timestamps)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA Out of Memory! Try shorter audio files or run on CPU.") from None
        finally:
            # Graceful cleanup of temp directory on Windows
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                logging.warning(f"Failed to clean up temp dir {tmp_dir}: {e}")

# =============================
# Main Application UI
# =============================

class AudioCensorApp:
    def __init__(self, root):
        self.root = root
        self.ui_lang_code = os.getenv("UI_LANGUAGE", "english").lower()
        self.t = ConfigLoader.load_ui_strings(self.ui_lang_code)
        
        if not self.t:
            messagebox.showerror("System Error", f"UI translation file 'ui-{self.ui_lang_code}.txt' not found!")
            sys.exit(1)

        self.root.title(APP_TITLE)
        self.root.geometry("700x620")
        self.root.resizable(False, False)

        default_lang = UI_LANG_MAP.get(self.ui_lang_code, "English")

        env_audio_path = os.getenv("AUDIO_FOLDER_PATH")
        self.audio_folder_var = tk.StringVar(value=env_audio_path if env_audio_path else "C:\\")
        self.gigaam_path = os.getenv("GIGAAM_MODEL_PATH", "ai-sage/GigaAM-Multilingual")
        self.parakeet_path = os.getenv("PARAKEET_MODEL_PATH", "nvidia/parakeet-tdt-0.6b-v3")


        self.model_var = tk.StringVar(value="Parakeet")
        self.whisper_ver_var = tk.StringVar(value="large-v3")
        self.language_var = tk.StringVar(value=default_lang)
        self.censor_lang_var = tk.StringVar(value=default_lang) 
        
        self.load_mode_var = tk.StringVar(value=self.t["mode_offline"])
        self.log_level_var = tk.StringVar(value=self.t["log_level_detailed"])
        self.censor_level = tk.IntVar(value=85)
        self.use_custom_sound = tk.BooleanVar(value=False)
        self.custom_sound: Optional[AudioSegment] = None
        self.custom_sound_path: str = ""
        self.censor_volume_db = tk.IntVar(value=-20)


        self.processing = False

        self.setup_ui()
        self.model_var.trace("w", self.on_model_change)
        self.on_model_change()
        self.validate_model_language()

    def validate_model_language(self):
        """Ensures the selected language is supported by the selected model."""
        model_name = self.model_var.get()
        current_lang = self.language_var.get()
        supported_langs = [name for name, code in LANGUAGE_OPTIONS[model_name]]
        if current_lang not in supported_langs:
            fallback = "English" if "English" in supported_langs else supported_langs[0]
            self.language_var.set(fallback)

    def setup_ui(self):
        """Initializes the GUI layout."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Audio Folder Selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(folder_frame, text=self.t["folder_label"]).pack(anchor="w")
        
        path_row = ttk.Frame(folder_frame)
        path_row.pack(fill="x", pady=2)
        self.folder_entry = ttk.Entry(path_row, textvariable=self.audio_folder_var, width=60)
        self.folder_entry.pack(side="left", fill="x", expand=True)
        self.browse_btn = ttk.Button(path_row, text=self.t["btn_browse"], command=self.select_audio_folder)
        self.browse_btn.pack(side="right", padx=(5, 0))


        # NN Model Selection
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=10)
        ttk.Label(top_frame, text=self.t["nn_label"]).grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, values=["GigaAM", "Whisper", "Parakeet"], state="readonly", width=15)
        self.model_combo.grid(row=0, column=1, padx=10, sticky="w")

        ttk.Label(top_frame, text=self.t["whisper_ver_label"]).grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.whisper_ver_combo = ttk.Combobox(top_frame, textvariable=self.whisper_ver_var, values=list(WHISPER_VERSION_MAP.keys()), state="readonly", width=12)
        self.whisper_ver_combo.grid(row=0, column=3, padx=10, sticky="w")

        # Mode and Language
        ttk.Label(top_frame, text=self.t["load_mode_label"]).grid(row=1, column=0, sticky="w", pady=10)
        self.mode_combo = ttk.Combobox(top_frame, textvariable=self.load_mode_var, values=[self.t["mode_online"], self.t["mode_offline"]], state="readonly", width=15)
        self.mode_combo.grid(row=1, column=1, padx=10, sticky="w")

        ttk.Label(top_frame, text=self.t["lang_label"]).grid(row=1, column=2, sticky="w", padx=(20, 0))
        self.lang_combo = ttk.Combobox(top_frame, textvariable=self.language_var, state="readonly", width=15)
        self.lang_combo.grid(row=1, column=3, padx=10, sticky="w")

        # Censoring Language
        censor_settings_frame = ttk.Frame(main_frame)
        censor_settings_frame.pack(fill="x", pady=10)
        ttk.Label(censor_settings_frame, text=self.t["censor_lang_label"]).grid(row=0, column=0, sticky="w")
        self.censor_lang_combo = ttk.Combobox(censor_settings_frame, textvariable=self.censor_lang_var, values=["Russian", "English"], state="readonly", width=15)
        self.censor_lang_combo.grid(row=0, column=1, padx=10, sticky="w")

        # Log Settings
        sys_frame = ttk.Frame(main_frame)
        sys_frame.pack(fill="x", pady=10)
        ttk.Label(sys_frame, text=self.t["log_label"]).grid(row=0, column=0, sticky="w")
        self.log_combo = ttk.Combobox(sys_frame, textvariable=self.log_level_var, values=[self.t["log_level_detailed"], self.t["log_level_basic"]], state="readonly", width=30)
        self.log_combo.grid(row=0, column=1, padx=10, sticky="w")
        self.log_combo.bind("<<ComboboxSelected>>", self.update_log_level)

        # Beep Intensity
        beep_frame = ttk.Frame(main_frame)
        beep_frame.pack(fill="x", pady=20)
        ttk.Label(beep_frame, text=self.t["beep_level_label"]).pack(side="left")
        ttk.Scale(beep_frame, from_=0, to=100, variable=self.censor_level, orient="horizontal", length=200).pack(side="left", padx=10)
        self.censor_label = ttk.Label(beep_frame, text="85%")
        self.censor_label.pack(side="left")
        self.censor_level.trace("w", lambda *a: self.censor_label.config(text=f"{self.censor_level.get()}%"))

        # Custom Sound Settings
        sound_frame = ttk.Frame(main_frame)
        sound_frame.pack(fill="x", pady=5)
        self.sound_checkbox = ttk.Checkbutton(sound_frame, text=self.t["custom_sound_label"], variable=self.use_custom_sound)
        self.sound_checkbox.pack(side="left")
        self.select_sound_btn = ttk.Button(sound_frame, text=self.t["btn_select_sound"], command=self.select_censor_sound)
        self.select_sound_btn.pack(side="left", padx=10)
        self.sound_label = ttk.Label(sound_frame, text=self.t["sound_default"])
        self.sound_label.pack(side="left")
        self.use_custom_sound.trace("w", lambda *a: self.on_sound_mode_change())


        # Volume Settings
        volume_frame = ttk.Frame(main_frame)
        volume_frame.pack(fill="x", pady=5)
        ttk.Label(volume_frame, text=self.t["volume_label"]).pack(side="left")
        self.volume_scale = ttk.Scale(volume_frame, from_=-40, to=0, variable=self.censor_volume_db, orient="horizontal", length=200)
        self.volume_scale.pack(side="left", padx=10)
        self.volume_label = ttk.Label(volume_frame, text="-20 dB")
        self.volume_label.pack(side="left")
        self.censor_volume_db.trace("w", lambda *a: self.volume_label.config(text=f"{self.censor_volume_db.get()} dB"))



        # Progress and Status
        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.pack(fill="x", pady=20)
        self.status_label = ttk.Label(main_frame, text=self.t["status_ready"])
        self.status_label.pack()

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        self.start_btn = ttk.Button(btn_frame, text=f"🚀 {self.t['btn_start']}", command=self.start_processing)
        self.start_btn.pack(side="left", padx=5)
        self.trans_btn = ttk.Button(btn_frame, text=f"📄 {self.t['btn_transcribe']}", command=lambda: self._run_thread(False))
        self.trans_btn.pack(side="left", padx=5)


    def select_audio_folder(self):
        """Opens a folder dialog to select the audio files directory."""
        selected = filedialog.askdirectory(title=self.t["folder_dialog_title"])
        if selected:
            self.audio_folder_var.set(selected)

    def update_log_level(self, event=None):
        """Updates the system logging level and outputs a localized message to the console."""
        val = self.log_level_var.get()
        level = logging.DEBUG if val == self.t["log_level_detailed"] else logging.INFO
        logging.getLogger().setLevel(level)
        
        # localized prefix for the change notification
        prefix = self.t.get("log_level_changed", "Console output changed to: ")
        logging.info(f"{prefix}{val}")

    def select_censor_sound(self):
        """Opens a file dialog to select a custom .wav sound for censoring."""
        file_path = filedialog.askopenfilename(
            title="Select Censor Sound",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        try:
            self.custom_sound = AudioSegment.from_wav(file_path)
            self.custom_sound_path = file_path
            filename = os.path.basename(file_path)
            duration_ms = len(self.custom_sound)
            self.sound_label.config(text=f"{filename} ({duration_ms/1000:.2f}s)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sound file:\n{e}")

    def on_sound_mode_change(self, *args):
        """Adjusts default volume when switching between custom sound and beep modes."""
        if self.use_custom_sound.get():
            self.censor_volume_db.set(-2)
        else:
            self.censor_volume_db.set(-20)


    def generate_censor_sound(self, duration_ms: int, censor_pct: float) -> AudioSegment:
        """Generates the censor sound — either custom .wav or default sine beep."""
        target_duration = int(duration_ms * censor_pct)
        volume_db = self.censor_volume_db.get()
        
        if self.use_custom_sound.get() and self.custom_sound is not None:
            if len(self.custom_sound) >= target_duration:
                sound = self.custom_sound[:target_duration]
            else:
                repeats = (target_duration // len(self.custom_sound)) + 1
                sound = (self.custom_sound * repeats)[:target_duration]
            return sound.apply_gain(volume_db)
        
        return pydub_gen.Sine(1000).to_audio_segment(target_duration).apply_gain(volume_db)



    def on_model_change(self, *args):
        """Updates the UI options based on the selected Neural Network model."""
        model_name = self.model_var.get()
        self.root.title(f"{APP_TITLE} ({model_name})")
        is_whisper = (model_name == "Whisper")
        state = "readonly" if is_whisper else "disabled"
        self.whisper_ver_combo.config(state=state)
        model_key = model_name
        langs = LANGUAGE_OPTIONS[model_key]
        self.lang_combo['values'] = [name for name, code in langs]
        if self.language_var.get() not in [name for name, code in langs]:
            self.language_var.set(langs[0][0])

    def get_lang_code(self):
        """Returns the ISO language code for the currently selected language."""
        model_key = self.model_var.get()
        for name, code in LANGUAGE_OPTIONS[model_key]:
            if name == self.language_var.get():
                return code
        return "ru"

    def _run_thread(self, full_process: bool):
        """Helper to run the processing logic in a background thread to keep GUI responsive."""
        if self.processing: return
        threading.Thread(target=self.process_files, args=(full_process,), daemon=True).start()

    def start_processing(self):
        self._run_thread(True)

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def update_progress(self, current_file_idx: int, total_files: int, current_sec: float, total_sec: float):
        if total_files == 0: return
        file_progress = (current_file_idx - 1) / total_files
        internal_progress = (current_sec / total_sec) / total_files if total_sec > 0 else 0
        total_pct = (file_progress + internal_progress) * 100
        self.root.after(0, lambda: self.progress.configure(value=total_pct))

    def process_files(self, full_process: bool):
        """Main processing loop for audio transcription and censoring."""
        self.processing = True
        self.root.after(0, lambda: self.start_btn.config(state="disabled"))
        
        # Validate existence of required lists
        if not os.path.exists("blacklist.txt") or not os.path.exists("whitelist.txt"):
            err_msg = self.t.get("error_missing_files", "Missing blacklist.txt or whitelist.txt")
            self.root.after(0, lambda: messagebox.showerror("Missing Files", err_msg))
            self.update_status(f"❌ {self.t['status_error']}")
            self.root.after(0, lambda: self.start_btn.config(state="normal"))
            self.processing = False
            return

        start_session_time = time.time()
        session_start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_type = self.model_var.get()
        version_str = self.whisper_ver_var.get() if model_type == "Whisper" else ""
        lang_code = self.get_lang_code()
        censor_pct = self.censor_level.get() / 100.0
        
        censor_lang_ui = self.censor_lang_var.get()
        mat_pattern = ConfigLoader.load_mat_regex(censor_lang_ui)

        audio_folder = self.audio_folder_var.get()
        stats_path = os.path.join(audio_folder, "stats.txt")
        log_path = os.path.join(audio_folder, "censor-log.txt")

        try:
            with open("blacklist.txt", encoding="utf-8") as f:
                blacklist = set(line.strip().lower() for line in f if line.strip())
            with open("whitelist.txt", encoding="utf-8") as f:
                whitelist = set(line.strip().lower() for line in f if line.strip())

            is_online = (self.load_mode_var.get() == self.t["mode_online"])
            self.update_status(self.t["status_loading"].format(model_type))
            
            # Initialize Model Engine
            if model_type == "Whisper":
                env_key = WHISPER_VERSION_MAP.get(version_str)
                source = version_str if is_online else os.getenv(env_key, version_str)
                engine = WhisperTranscriber(source)
            elif model_type == "Parakeet":
                source = "nvidia/parakeet-tdt-0.6b-v3" if is_online else self.parakeet_path
                engine = ParakeetTranscriber(source)

            else:
                source = "ai-sage/GigaAM-Multilingual" if is_online else self.gigaam_path
                engine = GigaAMTranscriber(source)
            
            engine.load_model()

            wav_files = [f for f in glob.glob(os.path.join(audio_folder, "*.wav")) if "_clean.wav" not in f]

            total_files = len(wav_files)

            with open(stats_path, "a", encoding="utf-8") as stats_f:
                stats_f.write(f"\n{self.t['log_session_start'].format(session_start_str, model_type)}\n")

            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(self.t["log_file_header"].format(session_start_str) + "\n\n")

            for idx, wav_path in enumerate(wav_files, 1):
                filename = os.path.basename(wav_path)
                self.update_status(self.t["status_processing"].format(idx, total_files, filename))
                
                audio = AudioSegment.from_wav(wav_path)
                total_audio_sec = len(audio) / 1000.0
                result_audio = AudioSegment.empty()
                last_end_ms = 0
                transcription_text = []
                badword_count = 0
                
                if model_type == "Whisper":
                    # Word-level processing for Whisper
                    segments = engine.transcribe(wav_path, lang_code)
                    for segment in segments:
                        self.update_progress(idx, total_files, segment.end, total_audio_sec)
                        timestamp = time.strftime('%H:%M:%S', time.gmtime(segment.start))
                        line = f"{timestamp}: {segment.text.strip()}"
                        transcription_text.append(line)
                        logging.info(f"{self.t['log_transcription']} {line}")

                        for word_obj in segment.words:
                            word_clean = re.sub(r'[^\w\s]', '', word_obj.word.lower()).strip()
                            if not word_clean: continue
                            if word_clean in whitelist: continue
                            if word_clean in blacklist or mat_pattern.search(word_clean):
                                badword_count += 1
                                with open(log_path, "a", encoding="utf-8") as lf:
                                    lf.write(self.t["log_file_entry"].format(filename, line, word_clean))
                                logging.debug(self.t["log_console_debug"].format(word_clean, timestamp))

                                if full_process:
                                    result_audio += audio[int(last_end_ms):int(word_obj.start * 1000)]
                                    duration_ms = int((word_obj.end - word_obj.start) * 1000)
                                    beep = self.generate_censor_sound(duration_ms, censor_pct)

                                    result_audio += beep
                                    last_end_ms = word_obj.end * 1000
                elif model_type == "Parakeet":
                    # Word-level processing for Parakeet with timestamps
                    result = engine.transcribe(wav_path, lang_code)
                    word_timestamps = result.timestamp.get("word", [])
                    
                    if not word_timestamps:
                        segment_timestamps = result.timestamp.get("segment", [])
                        for seg in segment_timestamps:
                            self.update_progress(idx, total_files, seg["end"], total_audio_sec)
                            timestamp = time.strftime('%H:%M:%S', time.gmtime(seg["start"]))
                            line = f"{timestamp}: {seg['segment'].strip()}"
                            transcription_text.append(line)
                            logging.info(f"{self.t['log_transcription']} {line}")
                            
                            words_in_text = re.findall(r'\b\w+\b', seg["segment"].lower())
                            found_bad = [w for w in words_in_text if (w in blacklist or mat_pattern.search(w)) and w not in whitelist]
                            if found_bad:
                                badword_count += len(found_bad)
                                for bw in found_bad:
                                    with open(log_path, "a", encoding="utf-8") as lf:
                                        lf.write(self.t["log_file_entry"].format(filename, line, bw))
                                    logging.debug(self.t["log_console_debug"].format(bw, timestamp))
                    else:
                        # First pass: collect valid words
                        valid_words = []
                        for word_ts in word_timestamps:
                            token = word_ts.get("token", "") or word_ts.get("text", "") or word_ts.get("word", "")
                            token_stripped = token.lstrip("|")
                            if not token_stripped:
                                continue
                            
                            word_clean = re.sub(r'[^\w\s]', '', token_stripped.lower()).strip()
                            if not word_clean:
                                continue
                            
                            valid_words.append({
                                "start": word_ts["start"],
                                "end": word_ts["end"],
                                "token": token_stripped,
                                "clean": word_clean
                            })
                        
                        # Group words into lines based on time gaps (< 0.5s = same line)
                        TIME_GAP_THRESHOLD = 0.5
                        current_line_words: List[Dict] = []
                        last_end_time = -1.0
                        
                        def flush_line():
                            nonlocal current_line_words, last_end_time, badword_count
                            if not current_line_words:
                                return
                            line_start = current_line_words[0]["start"]
                            timestamp = time.strftime('%H:%M:%S', time.gmtime(line_start))
                            text = " ".join(w["token"] for w in current_line_words)
                            line = f"{timestamp}: {text}"
                            transcription_text.append(line)
                            logging.info(f"{self.t['log_transcription']} {line}")
                            # Check each word in the line for profanity
                            for w in current_line_words:
                                if w["clean"] in whitelist:
                                    continue
                                if w["clean"] in blacklist or mat_pattern.search(w["clean"]):
                                    badword_count += 1
                                    with open(log_path, "a", encoding="utf-8") as lf:
                                        lf.write(self.t["log_file_entry"].format(filename, line, w["clean"]))
                                    logging.debug(self.t["log_console_debug"].format(w["clean"], timestamp))
                            current_line_words = []
                        
                        for w in valid_words:
                            self.update_progress(idx, total_files, w["end"], total_audio_sec)
                            if last_end_time >= 0 and (w["start"] - last_end_time) > TIME_GAP_THRESHOLD:
                                flush_line()
                            current_line_words.append(w)
                            last_end_time = w["end"]
                        
                        flush_line()  # Flush remaining words
                        
                        # Second pass: censor audio (beep replacement)
                        for w in valid_words:
                            if w["clean"] in whitelist:
                                continue
                            if w["clean"] in blacklist or mat_pattern.search(w["clean"]):
                                if full_process:
                                    result_audio += audio[int(last_end_ms):int(w["start"] * 1000)]
                                    duration_ms = int((w["end"] - w["start"]) * 1000)
                                    beep = self.generate_censor_sound(duration_ms, censor_pct)

                                    result_audio += beep
                                    last_end_ms = w["end"] * 1000

                else:
                    # Chunk-level processing for GigaAM
                    duration_ms = len(audio)
                    chunk_size = 25000
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for start_ms in range(0, duration_ms, chunk_size):
                            self.update_progress(idx, total_files, start_ms/1000.0, total_audio_sec)
                            end_ms = min(start_ms + chunk_size, duration_ms)
                            chunk = audio[start_ms:end_ms].set_frame_rate(16000).set_channels(1)
                            c_path = os.path.join(tmpdir, "chunk.wav")
                            chunk.export(c_path, format="wav")
                            res_text = str(engine.transcribe(c_path)).strip()
                            if res_text:
                                timestamp = time.strftime('%H:%M:%S', time.gmtime(start_ms/1000.0))
                                line = f"{timestamp}: {res_text}"
                                transcription_text.append(line)
                                logging.info(f"{self.t['log_transcription']} {line}")

                                words_in_text = re.findall(r'\b\w+\b', res_text.lower())
                                found_bad = [w for w in words_in_text if (w in blacklist or mat_pattern.search(w)) and w not in whitelist]
                                if found_bad:
                                    badword_count += len(found_bad)
                                    for bw in found_bad:
                                        with open(log_path, "a", encoding="utf-8") as lf:
                                            lf.write(self.t["log_file_entry"].format(filename, line, bw))
                                        logging.debug(self.t["log_console_debug"].format(bw, timestamp))

                                    if full_process:
                                        result_audio += audio[last_end_ms:start_ms]
                                        beep = self.generate_censor_sound(1000, 1.0)

                                        result_audio += beep
                                        last_end_ms = start_ms + 1000

                if full_process:
                    if last_end_ms < len(audio):
                        result_audio += audio[last_end_ms:]
                    if badword_count > 0:
                        clean_path = str(Path(wav_path).with_name(Path(wav_path).stem + "_clean.wav"))
                        result_audio = result_audio.set_frame_rate(audio.frame_rate)
                        result_audio = result_audio.set_channels(audio.channels)
                        result_audio.export(clean_path, format="wav", codec="pcm_s16le")


                # Export individual transcription file
                version_part = f"-{version_str}" if model_type == "Whisper" else ""
                trans_name = f"transcription{idx}-{Path(wav_path).stem}-{model_type}{version_part}.txt"
                with open(os.path.join(audio_folder, trans_name), "w", encoding="utf-8") as f:

                    f.write("\n".join(transcription_text))
                with open(stats_path, "a", encoding="utf-8") as stats_f:
                    stats_f.write(self.t["log_stats_case"].format(filename, badword_count) + "\n")

            end_session_time = time.time()
            elapsed = int(end_session_time - start_session_time)
            mins, secs = divmod(elapsed, 60)
            with open(stats_path, "a", encoding="utf-8") as stats_f:
                stats_f.write(self.t["log_session_end"].format(mins, secs) + "\n")
            
            logging.info(f"✅ Обработка завершена. Файлов: {total_files}, время: {mins}м {secs}с")

            self.update_status(f"✅ {self.t['status_done']}")
            messagebox.showinfo("Done", self.t["msg_done"].format(total_files))

        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            self.update_status(f"❌ {self.t['status_error']}")
            messagebox.showerror(self.t["msg_error"], str(e))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))
            self.root.after(0, lambda: self.progress.configure(value=0))

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioCensorApp(root)
    root.mainloop()
