import cv2
import time
import mediapipe as mp
import json
import sys
import os
import subprocess

if len(sys.argv) < 2:
	print("Usage : python video_input.py [youtube_video_url]")
	sys.exit(1)

youtube_url = sys.argv[1]
print(f"Téléchargement de la vidéo : {youtube_url}")

output_filename = "video.mp4"
command = [
	"yt-dlp",
	"-f", "best[ext=mp4]",
	"-o", output_filename,
	youtube_url
]

result = subprocess.run(command, capture_output=True, text=True)
if result.returncode != 0:
	print("Erreur lors du téléchargement de la vidéo :", result.stderr)
	sys.exit(1)


video_path = output_filename
#video_path = "videos/mediapipe_newmob_001.mp4" #only for local video.

capture = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(
	min_detection_confidence=0.3,
	min_tracking_confidence=0.3
)


#uncomment body parts that you want to display
selected_landmarks = {
 	# mp.solutions.pose.PoseLandmark.NOSE,
	# mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
	# mp.solutions.pose.PoseLandmark.LEFT_EYE,
	# mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER,
	# mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
	# mp.solutions.pose.PoseLandmark.RIGHT_EYE,
	# mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER,
	# mp.solutions.pose.PoseLandmark.LEFT_EAR,
	# mp.solutions.pose.PoseLandmark.RIGHT_EAR,
	# mp.solutions.pose.PoseLandmark.MOUTH_LEFT,
	# mp.solutions.pose.PoseLandmark.MOUTH_RIGHT,
	mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
	mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
	mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
	mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
	# mp.solutions.pose.PoseLandmark.LEFT_WRIST,
	# mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
	# mp.solutions.pose.PoseLandmark.LEFT_PINKY,
	# mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
	# mp.solutions.pose.PoseLandmark.LEFT_INDEX,
	# mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
	# mp.solutions.pose.PoseLandmark.LEFT_THUMB,
	# mp.solutions.pose.PoseLandmark.RIGHT_THUMB,
	mp.solutions.pose.PoseLandmark.LEFT_HIP,
	mp.solutions.pose.PoseLandmark.RIGHT_HIP,
	mp.solutions.pose.PoseLandmark.LEFT_KNEE,
	mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
	mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
	mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
	mp.solutions.pose.PoseLandmark.LEFT_HEEL,
	mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
	mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
	mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX
}

pose_connections = [
	(	mp.solutions.pose.PoseLandmark(a), 	mp.solutions.pose.PoseLandmark(b))
	for a, b in mp_pose.POSE_CONNECTIONS
	if 	mp.solutions.pose.PoseLandmark(a) in selected_landmarks and 	mp.solutions.pose.PoseLandmark(b) in selected_landmarks
]


previous_time = 0
frame_data = []
frame_count = 0

while capture.isOpened():
	ret, frame = capture.read()
	if not ret:
		break

	frame = cv2.resize(frame, (1280, 720))
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = pose_model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	if results.pose_landmarks:
		landmark_subset = {
			idx: lm for idx, lm in enumerate(results.pose_landmarks.landmark)
			if mp.solutions.pose.PoseLandmark(idx) in selected_landmarks
		}

		for a, b in pose_connections:
			if a.value in landmark_subset and b.value in landmark_subset:
				h, w, _ = image.shape
				pt1 = int(landmark_subset[a.value].x * w), int(landmark_subset[a.value].y * h)
				pt2 = int(landmark_subset[b.value].x * w), int(landmark_subset[b.value].y * h)
				cv2.line(image, pt1, pt2, (0, 255, 0), 2)
	
		for idx, lm in landmark_subset.items():
			h, w, _ = image.shape
			cx, cy = int(lm.x * w), int(lm.y * h)
			cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

		frame_landmarks = {"frame": frame_count}
		for mark, data_point in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
			frame_landmarks[mark.name] = {
				'x': data_point.x,
				'y': data_point.y,
				'z': data_point.z,
				'visibility': data_point.visibility
			}
		frame_data.append(frame_landmarks)

 
	current_time = time.time()
	fps = 1 / (current_time - previous_time)
	previous_time = current_time
	cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

	cv2.imshow("Filtered Pose Skeleton", image)

	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

	frame_count += 1

with open("landmarks_data.txt", "w") as f:
	for frame in frame_data:
		f.write(f"Frame {frame['frame']}:\n")
		for k, v in frame.items():
			if k != "frame":
				f.write(f"{k}: {v}\n")
		f.write("\n")

capture.release()
cv2.destroyAllWindows()