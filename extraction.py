import cv2
import os

def extract_frames_from_folder(videos_folder, output_folder, frame_interval=10, max_frames_per_video=None):
    # Get a list of all video files in the folder
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0  # Global frame counter to ensure unique filenames

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_index = 0  # To track the frame number within the current video
        extracted_frames = 0  # Count of frames extracted from the current video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames_per_video and extracted_frames >= max_frames_per_video):
                break

            # Save the frame only if it's at the specified interval
            if frame_index % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
                extracted_frames += 1  # Keep track of how many frames are saved for this video

            frame_index += 1

        cap.release()
        print(f"Extracted {extracted_frames} frames from {video_file}")

# Example usage:
# Extract every 10th frame, but no more than 100 frames per video
extract_frames_from_folder(r'C:\Users\USER\Desktop\Deepfake_project\val\fake', r'C:\Users\USER\Desktop\Deepfake_project\val1\fake', frame_interval=10, max_frames_per_video=100)