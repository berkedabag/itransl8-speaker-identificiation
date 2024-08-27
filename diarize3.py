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
from transcription_helpers import transcribe_batched, transcribe
import subprocess

class AudioProcessor:
    def __init__(self, audio_path, output_path, language, device="cuda", model_name="large-v3", batch_size=16, suppress_numerals=False, stemming=True):
        self.audio_path = audio_path
        self.output_path = output_path
        self.language = language
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        self.batch_size = batch_size
        self.suppress_numerals = suppress_numerals
        self.stemming = stemming
        self.compute_dtype = 'float16' if self.device == 'cuda' else 'float32'  # Set compute_dtype based on device

    def process_audio(self):
        try:
            vocal_target = self.separate_audio() if self.stemming else self.audio_path
            whisper_results, language = self.transcribe_audio(vocal_target)
            word_timestamps = self.handle_language_specific_tasks(whisper_results, language, vocal_target)
            self.convert_to_mono(vocal_target)
            self.perform_diarization()
            txt_path, srt_path = self.diarization_results(word_timestamps)
            self.cleanup()
            return txt_path, srt_path
        except Exception as e:
            logging.info(f"An error occurred: {e}")
            return None, None

    def separate_audio(self):
        demucs_model = "htdemucs_ft"
        # Use the Python interpreter from the current virtual environment
        python_interpreter = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv/bin/python3')
        
        command = [
            python_interpreter, '-m', 'demucs.separate',
            '-n', demucs_model,
            '--two-stems=vocals',
            self.audio_path,
            '-o', self.output_path
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info("Source splitting succeeded.")
            logging.debug(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Source splitting failed with error: {e.stderr}. Using original audio file.")
            return self.audio_path
        except Exception as e:
            logging.error(f"Unexpected error during source splitting: {e}")
            return self.audio_path

        return os.path.join(self.output_path, demucs_model, os.path.splitext(os.path.basename(self.audio_path))[0], "vocals.wav")

    def transcribe_audio(self, vocal_target):
        transcribe_function = transcribe_batched if self.batch_size != 0 else transcribe
        return transcribe_function(
            vocal_target, 
            self.language, 
            self.batch_size, 
            self.model_name, 
            self.compute_dtype,  # Added compute_dtype
            suppress_numerals=self.suppress_numerals, 
            device=self.device
        )   
    def handle_language_specific_tasks(self, whisper_results, language, vocal_target):
        if language in wav2vec2_langs:
            alignment_model, metadata = whisperx.load_align_model(language_code=language, device=self.device)
            result_aligned = whisperx.align(whisper_results, alignment_model, metadata, vocal_target, self.device)
            word_timestamps = filter_missing_timestamps(result_aligned["word_segments"], initial_timestamp=whisper_results[0].get("start"), final_timestamp=whisper_results[-1].get("end"))
            del alignment_model
            torch.cuda.empty_cache()
            return word_timestamps
        else:
            assert self.batch_size == 0, "Unsupported language, set --batch_size to 0."
            word_timestamps = [{"word": word[2], "start": word[0], "end": word[1]} for segment in whisper_results for word in segment["words"]]
            return word_timestamps
        
    def convert_to_mono(self, vocal_target):
        sound = AudioSegment.from_file(vocal_target).set_channels(1)
        root = os.getcwd()
        sound.export(os.path.join(self.output_path, "mono_file.wav"), format="wav")

    def perform_diarization(self):
        msdd_model = NeuralDiarizer(cfg=create_config(self.output_path)).to(self.device)
        msdd_model.diarize()
        del msdd_model
        torch.cuda.empty_cache()

    def diarization_results(self, word_timestamps):
        speaker_ts = []
        with open(os.path.join(self.output_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
        #print(speaker_ts)
        #print(word_timestamps)
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if self.language in punct_model_langs:
            # restoring punctuation in the transcript to help realign the sentences
            try:
                punct_model = PunctuationModel(model="kredor/punctuate-all")

                words_list = list(map(lambda x: x["word"], wsm))

                labled_words = punct_model.predict(words_list)

                ending_puncts = ".?!"
                model_puncts = ".,;:!?"

                # We don't want to punctuate U.S.A. with a period. Right?
                is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

                for word_dict, labeled_tuple in zip(wsm, labled_words):
                    word = word_dict["word"]
                    if (
                        word
                        and labeled_tuple[1] in ending_puncts
                        and (word[-1] not in model_puncts or is_acronym(word))
                    ):
                        word += labeled_tuple[1]
                        if word.endswith(".."):
                            word = word.rstrip(".")
                        word_dict["word"] = word
            except Exception as e:
                logging.warning(f"Failed to restore punctuation: {e}")
                

        else:
            logging.warning(
                f"Punctuation restoration is not available for {self.language} language. Using the original punctuation."
            )
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        txt_output_path = os.path.join(self.output_path, f"{os.path.splitext(os.path.basename(self.audio_path))[0]}.txt")
        unwanted_words = {'Altyazı M.K.', 'Subtitle M.K.'}
        with open(txt_output_path, "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f, unwanted_words)

        srt_output_path = os.path.join(self.output_path, f"{os.path.splitext(os.path.basename(self.audio_path))[0]}.srt")
        with open(srt_output_path, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt, unwanted_words)

        return txt_output_path, srt_output_path

    def cleanup(self):
        # Clean up any temporary files or data
        pass

"""

if __name__ == "__main__":
    audio_processor = AudioProcessor(
        audio_path="/home/king/Desktop/berke/itransl8-API/sub_systems/speaker_identification/example.mp3",
        output_path="/home/king/Desktop/berke/itransl8-API/sub_systems/speaker_identification/exampleoutput",
        language="en",
        device="cuda"
    )
    audio_processor.process_audio() """