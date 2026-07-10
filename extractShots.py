import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import librosa
import numpy as np
import tempfile
import ffmpeg
import uuid
import glob
import sys

def detect_tennis_shots(audio_path):
    """Detect tennis shot timestamps from audio file"""
    temp_wav_path = None
    try:
        temp_wav_handle, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_wav_handle)
        
        stream = (
            ffmpeg
            .input(audio_path)
            .output(
                temp_wav_path,
                acodec='pcm_s16le',
                ac=1,
                ar=22050,
                loglevel='error'
            )
            .overwrite_output()
        )
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        
        y, sr = librosa.load(temp_wav_path, sr=22050)
        y = librosa.util.normalize(y)
        
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            hop_length=512,
            lag=2,
            max_size=4,
            center=True,
            aggregate=np.median
        )
        
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            wait=10,
            pre_avg=25,
            post_avg=25,
            delta=0.35,
            backtrack=True
        )
        
        timestamps = librosa.frames_to_time(onset_frames, sr=sr)
        
        filtered_timestamps = []
        last_timestamp = -1
        for ts in timestamps:
            if last_timestamp == -1 or (ts - last_timestamp) >= 0.6:
                filtered_timestamps.append(ts)
                last_timestamp = ts
        
        return np.array(filtered_timestamps)
        
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

def extract_shot_clips(video_path, timestamps, output_dir, pre_shot=0.5, post_shot=0.7):
    """Extract short video clips around individual tennis shots, downsample to 25fps"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    duration = float(probe['format']['duration'])
    
    output_fps = 25

    print(f"Video properties:")
    print(f"- Original FPS: {eval(video_info['r_frame_rate'])}")
    print(f"- Resolution: {video_info['width']}x{video_info['height']}")
    print(f"- Duration: {duration:.2f}s")
    print(f"- Output FPS: {output_fps}")
    print(f"- Clip duration: {pre_shot + post_shot:.2f}s")
    
    clips_written = 0
    for i, timestamp in enumerate(timestamps):
        start_time = max(0, timestamp - pre_shot)
        end_time = min(duration, timestamp + post_shot)
        clip_duration = end_time - start_time
        
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{unique_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            stream = (
                ffmpeg
                .input(video_path, ss=start_time)
                .output(
                    output_path,
                    t=clip_duration,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate='5000k',
                    audio_bitrate='128k',
                    preset='fast',
                    r=output_fps,
                    vsync='cfr',
                    movflags='+faststart',
                    loglevel='error'
                )
                .overwrite_output()
            )
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                clips_written += 1
                print(f"Clip {i + 1} written: {output_path}")
        except Exception as e:
            print(f"Error processing clip {i + 1}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
    
    print(f"Total clips written: {clips_written}/{len(timestamps)}")

def process_training_data(base_dir):
    """Process training data from separate audio and video files"""
    recordings_dir = os.path.join(base_dir, "Full Recording")
    output_dir = os.path.join(base_dir, "Clips")
    os.makedirs(output_dir, exist_ok=True)

    for match_dir in glob.glob(os.path.join(recordings_dir, "*")):
        if not os.path.isdir(match_dir):
            continue
        
        print(f"\nProcessing match: {os.path.basename(match_dir)}")
        
        audio_file = next((f for f in glob.glob(os.path.join(match_dir, "*.mp3"))), None)
        video_file = next((f for f in glob.glob(os.path.join(match_dir, "*.mp4"))), None)
        
        if not audio_file or not video_file:
            print("Missing audio or video file. Skipping match.")
            continue
        
        print("Detecting tennis shots...")
        timestamps = detect_tennis_shots(audio_file)
        print(f"Found {len(timestamps)} shots")
        
        if len(timestamps) == 0:
            print("No shots detected. Skipping match.")
            continue
        
        print("Extracting video clips...")
        extract_shot_clips(video_file, timestamps, output_dir, pre_shot=0.8, post_shot=1.0)

def main():
    base_dir = "Data"
    process_training_data(base_dir)

if __name__ == "__main__":
    main()