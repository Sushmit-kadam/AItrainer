import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Main Streamlit app logic
def main():
    # Your Streamlit app content goes here
    st.title("Bicep Curl Counter with Pose Estimation")
    st.write("Real-time curl counter with form feedback.")
    counter_text = st.empty()

    def calculate_angle(a, b, c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    # Previous angle for speed evaluation
    prev_angle = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Assess elbow and shoulder alignment
                shoulder_x = shoulder[0]
                elbow_x = elbow[0]
                if shoulder_x > elbow_x:
                    # Elbow is not aligned with the shoulder
                    feedback_message = "Elbow Not Aligned"
                    feedback_color = (0, 0, 255)
                else:
                    feedback_message = "Good Form"
                    feedback_color = (0, 255, 0)

                # Assess repetition speed
                rep_speed_feedback = ""
                if prev_angle is not None:
                    speed = abs(angle - prev_angle)
                    if speed > 30:
                        rep_speed_feedback = "Slow down!"
                        feedback_color = (0, 0, 255)

                # Update previous angle
                prev_angle = angle

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print("Rep Count:", counter)
                    counter_text.markdown(f"**Reps:** {counter}")




            except:
                pass

            # Render curl counter and feedback
            cv2.rectangle(image, (0, 0), (225, 123), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback data
            cv2.putText(image, feedback_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 1, cv2.LINE_AA)
            cv2.putText(image, rep_speed_feedback, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

