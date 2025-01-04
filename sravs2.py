import cv2
import mediapipe as mp
import time
import os
import threading
import wave
from moviepy.editor import VideoFileClip, AudioFileClip
import pyaudio

# Create 'captures' directory if it doesn't exist
if not os.path.exists('captures'):
    os.makedirs('captures')

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Function to record audio
def record_audio(audio_filename, stop_event):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to a file
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to merge video and audio
def merge_video_audio(video_filename, audio_filename, output_filename):
    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)
    video_with_audio = video_clip.set_audio(audio_clip)
    video_with_audio.write_videofile(output_filename, codec="libx264", audio_codec="aac")

# Function to count fingers
def count_fingers(hand_landmarks):
    fingers = [0] * 5
    fingers[0] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    for i, tip in enumerate([mp_hands.HandLandmark.INDEX_FINGER_TIP,
                              mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                              mp_hands.HandLandmark.RING_FINGER_TIP,
                              mp_hands.HandLandmark.PINKY_TIP]):
        fingers[i + 1] = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
    return sum(fingers)

# Function to detect gestures
def detect_gesture(landmarks):
    if (landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[6].y and
        landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y):
        return "start"
    if (landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[6].y and
        landmarks[20].y < landmarks[18].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y):
        return "stop"
    return None

# Initialize webcam
cap = cv2.VideoCapture(0)
is_recording = False
output_video = None
capturing = False
timer_start = None
timer_duration = 0
cooldown_start = None
cooldown_duration = 3
audio_thread = None
audio_stop_event = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cooldown_active = cooldown_start and (time.time() - cooldown_start < cooldown_duration)
    remaining_cooldown = cooldown_duration - (time.time() - cooldown_start) if cooldown_active else 0

    if cooldown_active:
        status_text = f"Cooldown: {int(remaining_cooldown)}s"
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    result = hands.process(rgb_frame)
    total_fingers = 0
    captured_frame = frame.copy()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            if not is_recording:
                total_fingers += count_fingers(hand_landmarks)

            gesture = detect_gesture(hand_landmarks.landmark)

            if gesture == "start" and not is_recording:
                is_recording = True
                capturing = False  # Stop capturing images during video recording
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                timestamp = int(time.time())
                video_filename = f'captures/raw_video_{timestamp}.avi'
                audio_filename = f'captures/audio_{timestamp}.wav'
                output_video_filename = f'captures/output_video_with_audio_{timestamp}.mp4'
                output_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

                audio_stop_event = threading.Event()
                audio_thread = threading.Thread(target=record_audio, args=(audio_filename, audio_stop_event))
                audio_thread.start()
                print("Recording started")

            elif gesture == "stop" and is_recording:
                is_recording = False
                output_video.release()
                audio_stop_event.set()
                audio_thread.join()
                merge_video_audio(video_filename, audio_filename, output_video_filename)
                print(f"Recording stopped and video saved: {output_video_filename}")
                cooldown_start = time.time()

           # mp_drawing.draw_landmarks(captured_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if not is_recording:
        if total_fingers > 0 and not capturing:
            capturing = True
            timer_start = time.time()
            timer_duration = total_fingers + 1

        if capturing:
            elapsed_time = time.time() - timer_start
            countdown = int(timer_duration - elapsed_time)
            if countdown > 0:
                cv2.putText(frame, f'Capturing in {countdown}s', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                filename = f'captures/captured_image_{int(time.time())}.jpg'
                cv2.imwrite(filename, captured_frame)
                print(f"Image saved: {filename}")
                capturing = False
                cooldown_start = time.time()

    if is_recording:
        output_video.write(frame)

    status_text = "Recording..." if is_recording else f'Fingers: {total_fingers}'
    cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_recording else (255, 255, 255), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if output_video:
    output_video.release()
cap.release()
cv2.destroyAllWindows()
