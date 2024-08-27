import uuid
import requests
import time
import os
import logging
from diarize3 import AudioProcessor
import boto3

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_S3_BUCKET_NAME = 'video-ai-dubbing'
REGION_NAME = 'eu-central-1'
server_ip = "https://itransl8.com:8000"
api_key = ""

# Setup logging
logging.basicConfig(filename='speaker_identification.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',  filemode='w')

s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=REGION_NAME)


def download_from_s3(s3_key, local_path, bucket_name):
    try:
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        s3_client.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Successfully downloaded {s3_key} to {local_path}.")
    except Exception as e:
        logging.error(
            f"An error occurred while downloading {s3_key} from S3: {e}")


def upload_to_s3(file_path, bucket_name, s3_file_path):
    aws_active = True
    try:
        if aws_active:
            s3_client.upload_file(file_path, bucket_name, s3_file_path)
            logging.info(
                f"Uploaded {file_path} to S3 bucket {bucket_name} successfully.")
        else:
            logging.info(
                f"Fake uploaded {file_path} to S3 bucket {bucket_name}.")
            time.sleep(10)
    except Exception as e:
        logging.error(f"An error occurred while uploading to S3: {e}")


def update_dubbing_status(dubbing_id, new_status):
    update_url = f"{server_ip}/update-dubbing-status/{dubbing_id}"
    headers = {'x-api-key': api_key}
    data = {'status': new_status}
    try:
        response = requests.post(update_url, json=data, headers=headers)
        if response.status_code == 200:
            logging.info(
                f"Successfully updated status for dubbing {dubbing_id} to {new_status}.")
        else:
            logging.error(
                f"Failed to update status for dubbing {dubbing_id}: {response.status_code}")
    except Exception as e:
        logging.error(f"Failed to update status due to an error: {e}")


def fetch_dubbings(status):
    base_url = f"{server_ip}/dubbings-by-status/"
    headers = {'x-api-key': api_key}
    try:
        response = requests.get(f"{base_url}{status}", headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            logging.info("No more dubbings available, sleeping...")
            time.sleep(10)
            return []
        else:
            logging.error("Failed to fetch dubbings: %s", response.status_code)
            time.sleep(10)
            return []
    except Exception as e:
        logging.error(f"Failed to fetch dubbings due to an error: {e}")
        return []


def clean_up(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clean_up(file_path)
        os.rmdir(directory)
        logging.info(f"Cleaned up directory {directory}")
    except Exception as e:
        logging.error(f"An error occurred while cleaning up: {e}")


status_to_query = "video_downloaded"

while True:
    logging.info("Fetching dubbings...")
    dubbings = fetch_dubbings(status_to_query)
    for dubbing in dubbings:
        dubbing_id = dubbing['dubbing_id']
        input_lang = dubbing['input_lang']
        output_lang = dubbing['output_lang']
        logging.info(f"Downloading content for dubbing_id: {dubbing_id}...")
        local_uuid = uuid.uuid4()
        s3_path = f"dubbings/{dubbing_id}/audio.mp3"
        local_temp_path = f"temp/{local_uuid}"
        local_path = f"{local_temp_path}/audio.mp3"

        download_from_s3(s3_key=s3_path, local_path=local_path,
                         bucket_name=AWS_S3_BUCKET_NAME)
        logging.info("diarization started...")
        start_time = time.time()
        diarize_system = AudioProcessor(
            audio_path=local_path, output_path=local_temp_path, language=input_lang)
        txt_path, srt_path = diarize_system.process_audio()
        vocals_path = local_temp_path + '/htdemucs_ft/audio/vocals.wav'
        non_vocals_path = local_temp_path + '/htdemucs_ft/audio/no_vocals.wav'
        logging.info("diarization process finished... time taken: %s",
                     time.time()-start_time)
        logging.info(
            f"Uploading video and audio for dubbing_id: {dubbing_id} to S3...")
        upload_to_s3(srt_path, AWS_S3_BUCKET_NAME,
                     f"dubbings/{dubbing_id}/video.srt")
        upload_to_s3(txt_path, AWS_S3_BUCKET_NAME,
                     f"dubbings/{dubbing_id}/video.txt")
        upload_to_s3(vocals_path, AWS_S3_BUCKET_NAME,
                     f"dubbings/{dubbing_id}/vocals.wav")
        upload_to_s3(non_vocals_path, AWS_S3_BUCKET_NAME,
                     f"dubbings/{dubbing_id}/no_vocals.wav")
        logging.info(f"Updating status for dubbing_id: {dubbing_id}...")
        update_dubbing_status(dubbing['dubbing_id'], "speech2text")
        logging.info(f"Cleaning up for unique_id: {local_uuid}...")
        # clean_up(local_temp_path)
