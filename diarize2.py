import argparse
import os
import re
import logging
import torch
from pydub import AudioSegment
from helpers import *
from faster_whisper import WhisperModel
import whisperx
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from transcription_helpers import transcribe_batched

class AudioProcessor:
    def __init__(self):
        self.mtypes = {"cpu": "int8", "cuda": "float16"}
        self.args = self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", "--audio", help="name of the target audio file", required=True)
        parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True, help="Disables source separation.")
        parser.add_argument("--suppress_numerals", action="store_true", dest="suppress_numerals", default=False, help="Suppresses Numerical Digits.")
        parser.add_argument("--whisper-model", dest="model_name", default="medium.en", help="name of the Whisper model to use")
        parser.add_argument("--batch-size", type=int, dest="batch_size", default=16, help="Batch size for batched inference")
        parser.add_argument("--language", type=str, default=None, choices=whisper_langs, help="Language spoken in the audio")
        parser.add_argument("--device", dest="device", default="cuda" if torch.cuda.is_available() else "cpu", help="Processing device")
        return parser.parse_args()

    def process_audio(self):
        vocal_target = self.separate_audio() if self.args.stemming else self.args.audio
        whisper_results, language = self.transcribe_audio(vocal_target)
        self.handle_language_specific_tasks(whisper_results, language, vocal_target)
        self.convert_to_mono(vocal_target)
        self.perform_diarization()
        self.cleanup()

    def separate_audio(self):
        demucs_model = "htdemucs_ft"
        return_code = os.system(f'python3 -m demucs.separate -n {demucs_model} --two-stems=vocals "{self.args.audio}" -o "temp_outputs"')
        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file.")
            return self.args.audio
        return os.path.join("temp_outputs", demucs_model, os.path.splitext(os.path.basename(self.args.audio))[0], "vocals.wav")

    def transcribe_audio(self, vocal_target):
        transcribe_function = transcribe_batched if self.args.batch_size != 0 else transcribe
        return transcribe_function(vocal_target, self.args.language, self.args.batch_size, self.args.model_name, self.mtypes[self.args.device], self.args.suppress_numerals, self.args.device)

    def handle_language_specific_tasks(self, whisper_results, language, vocal_target):
        if language in wav2vec2_langs:
            alignment_model, metadata = whisperx.load_align_model(language_code=language, device=self.args.device)
            result_aligned = whisperx.align(whisper_results, alignment_model, metadata, vocal_target, self.args.device)
            word_timestamps = filter_missing_timestamps(result_aligned["word_segments"], initial_timestamp=whisper_results[0].get("start"), final_timestamp=whisper_results[-1].get("end"))
            del alignment_model
            torch.cuda.empty_cache()
        else:
            assert self.args.batch_size == 0, "Unsupported language, set --batch_size to 0."
            word_timestamps = [{"word": word[2], "start": word[0], "end": word[1]} for segment in whisper_results for word in segment["words"]]

    def convert_to_mono(self, vocal_target):
        sound = AudioSegment.from_file(vocal_target).set_channels(1)
        root = os.getcwd()
        temp_path = os.path.join(root, "temp_outputs")
        os.makedirs(temp_path, exist_ok=True)
        sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

    def perform_diarization(self):
        msdd_model = NeuralDiarizer(cfg=create_config("temp_outputs")).to(self.args.device)
        msdd_model.diarize()
        del msdd_model
        torch.cuda.empty_cache()

    def cleanup(self):
        # Clean up any temporary files or data
        pass

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    audio_processor.process_audio()
