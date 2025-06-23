import math
import os
import queue
import tempfile
import time
import warnings

from prompt_toolkit.shortcuts import prompt

from aider.llm import litellm

from .dump import dump  # noqa: F401

try:
    import whisper
    LOCAL_WHISPER_AVAILABLE = True
except ImportError:
    LOCAL_WHISPER_AVAILABLE = False
    whisper = None

warnings.filterwarnings(
    "ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)
warnings.filterwarnings("ignore", category=SyntaxWarning)


from pydub import AudioSegment  # noqa
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError  # noqa

try:
    import soundfile as sf
except (OSError, ModuleNotFoundError):
    sf = None


class SoundDeviceError(Exception):
    pass


class Voice:
    max_rms = 0
    min_rms = 1e5
    pct = 0

    threshold = 0.15

    def __init__(self, audio_format="wav", device_name=None):
        if sf is None:
            raise SoundDeviceError
        try:
            print("Initializing sound device...")
            import sounddevice as sd

            self.sd = sd

            devices = sd.query_devices()

            if device_name:
                # Find the device with matching name
                device_id = None
                for i, device in enumerate(devices):
                    if device_name in device["name"]:
                        device_id = i
                        break
                if device_id is None:
                    available_inputs = [d["name"] for d in devices if d["max_input_channels"] > 0]
                    raise ValueError(
                        f"Device '{device_name}' not found. Available input devices:"
                        f" {available_inputs}"
                    )

                print(f"Using input device: {device_name} (ID: {device_id})")

                self.device_id = device_id
            else:
                self.device_id = None

        except (OSError, ModuleNotFoundError):
            raise SoundDeviceError
        if audio_format not in ["wav", "mp3", "webm"]:
            raise ValueError(f"Unsupported audio format: {audio_format}")
        self.audio_format = audio_format
        
        # Initialize local whisper model if available
        self.local_model = None
        if LOCAL_WHISPER_AVAILABLE:
            try:
                print("Loading local Whisper small model...")
                self.local_model = whisper.load_model("small")
                print("Local Whisper model loaded successfully!")
            except Exception as e:
                print(f"Failed to load local Whisper model: {e}")
                self.local_model = None

    def transcribe_local(self, filename, language="en", history=""):
        """Transcribe audio using local Whisper model."""
        if not self.local_model:
            return None
            
        try:
            print("Transcribing with local Whisper...")
            # Check if file exists
            if not os.path.exists(filename):
                print(f"Audio file not found: {filename}")
                return None
            
            # Get file size for debugging
            file_size = os.path.getsize(filename)
            print(f"Audio file: {filename} ({file_size} bytes)")
            
            result = self.local_model.transcribe(filename, language=language)
            text = result["text"].strip()
            print(f"Transcribed: {text}")
            return text
        except FileNotFoundError as e:
            if "ffmpeg" in str(e) or "[WinError 2]" in str(e):
                print("FFmpeg not found in PATH. Please install FFmpeg:")
                print("  Windows: choco install ffmpeg  OR  winget install ffmpeg")
                print("  macOS: brew install ffmpeg")
                print("  Linux: sudo apt install ffmpeg")
                return None
            else:
                print(f"File not found error: {e}")
                return None
        except Exception as e:
            print(f"Local transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        import numpy as np

        rms = np.sqrt(np.mean(indata**2))
        self.max_rms = max(self.max_rms, rms)
        self.min_rms = min(self.min_rms, rms)

        rng = self.max_rms - self.min_rms
        if rng > 0.001:
            self.pct = (rms - self.min_rms) / rng
        else:
            self.pct = 0.5

        self.q.put(indata.copy())

    def get_prompt(self):
        num = 10
        if math.isnan(self.pct) or self.pct < self.threshold:
            cnt = 0
        else:
            cnt = int(self.pct * 10)

        bar = "░" * cnt + "█" * (num - cnt)
        bar = bar[:num]

        dur = time.time() - self.start_time
        return f"Recording, press ENTER when done... {dur:.1f}sec {bar}"

    def record_and_transcribe(self, history=None, language=None):
        try:
            return self.raw_record_and_transcribe(history, language)
        except KeyboardInterrupt:
            return
        except SoundDeviceError as e:
            print(f"Error: {e}")
            print("Please ensure you have a working audio input device connected and try again.")
            return

    def raw_record_and_transcribe(self, history, language):
        self.q = queue.Queue()

        temp_wav = tempfile.mktemp(suffix=".wav")

        try:
            sample_rate = int(self.sd.query_devices(self.device_id, "input")["default_samplerate"])
        except (TypeError, ValueError):
            sample_rate = 16000  # fallback to 16kHz if unable to query device
        except self.sd.PortAudioError:
            raise SoundDeviceError(
                "No audio input device detected. Please check your audio settings and try again."
            )

        self.start_time = time.time()

        try:
            with self.sd.InputStream(
                samplerate=sample_rate, channels=1, callback=self.callback, device=self.device_id
            ):
                prompt(self.get_prompt, refresh_interval=0.1)
        except self.sd.PortAudioError as err:
            raise SoundDeviceError(f"Error accessing audio input device: {err}")

        with sf.SoundFile(temp_wav, mode="x", samplerate=sample_rate, channels=1) as file:
            while not self.q.empty():
                file.write(self.q.get())

        use_audio_format = self.audio_format

        # Check file size and offer to convert to mp3 if too large
        file_size = os.path.getsize(temp_wav)
        if file_size > 24.9 * 1024 * 1024 and self.audio_format == "wav":
            print("\nWarning: {temp_wav} is too large, switching to mp3 format.")
            use_audio_format = "mp3"

        filename = temp_wav
        if use_audio_format != "wav":
            try:
                new_filename = tempfile.mktemp(suffix=f".{use_audio_format}")
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(new_filename, format=use_audio_format)
                os.remove(temp_wav)
                filename = new_filename
            except (CouldntDecodeError, CouldntEncodeError) as e:
                print(f"Error converting audio: {e}")
            except (OSError, FileNotFoundError) as e:
                print(f"File system error during conversion: {e}")
            except Exception as e:
                print(f"Unexpected error during audio conversion: {e}")

        # Try local transcription first, fallback to remote only if no local model
        text = None
        if self.local_model:
            text = self.transcribe_local(filename, language=language, history=history)
            if not text:
                print("Local transcription returned empty result")
        else:
            print("No local model available, using remote transcription")
            # Fallback to remote transcription only if no local model
            with open(filename, "rb") as fh:
                try:
                    transcript = litellm.transcription(
                        model="whisper-1", file=fh, prompt=history, language=language
                    )
                    text = transcript.text
                except Exception as err:
                    print(f"Unable to transcribe {filename}: {err}")
                    if filename != temp_wav:
                        os.remove(filename)
                    return

        if filename != temp_wav:
            os.remove(filename)
        return text


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    print(Voice().record_and_transcribe())
