import random
import time
import cv2
import mediapipe as mp
from pose_comparator import compare_poses, load_pose_model
from pose_extractor import extract_pose_from_image

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def capture_image(filename='current_frame.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return False

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        cap.release()
        return True
    else:
        print("Error: Could not capture image.")
        cap.release()
        return False

def load_all_models():
    return {
        'archery': ['models/archery1.jpg.pkl', 'models/archery2.jpg.pkl', 'models/archery3.jpg.pkl', 'models/archery3_left.jpg.pkl', 'models/archery4.mp4.pkl'],
        'basketball': ['models/basketball1.jpeg.pkl', 'models/basketball2.jpeg.pkl', 'models/basketball3.jpg.pkl', 'models/basketball4.jpg.pkl', 'models/basketball5.mp4.pkl', 'models/basketball6.mp4.pkl', 'models/basketball7.mp4.pkl'],
        'boxing': ['models/boxing1.jpg.pkl', 'models/boxing1_left.jpg.pkl', 'models/boxing2.png.pkl', 'models/boxing2_left.png.pkl', 'models/boxing3.jpeg.pkl', 'models/boxing3_left.jpeg.pkl', 'models/boxing4.jpeg.pkl', 'models/boxing5.jpeg.pkl', 'models/boxing6.mp4.pkl', 'models/boxing7.mp4.pkl', 'models/boxing8.mp4.pkl', 'models/boxing9.mp4.pkl', 'models/boxing10.mp4.pkl'], 
        'golf': ['models/golf1.webp.pkl', 'models/golf1_left.jpg.pkl', 'models/golf2.jpg.pkl', 'models/golf2_left.jpg.pkl', 'models/golf3_left.jpeg.pkl', 'models/golf4.mp4.pkl', 'models/golf5.mp4.pkl'],
        'Hurray': ['models/Hurray1.jpeg.pkl', 'models/Hurray2.jpeg.pkl', 'models/Hurray3.mp4.pkl', 'models/Hurray4.jpeg.pkl', 'models/Hurray5.jpeg.pkl', 'models/Hurray6.jpeg.pkl'],
        'table_tennis': ['models/table_ tennis1.jpg.pkl', 'models/table_ tennis1_left.jpg.pkl', 'models/table_tennis2.jpeg.pkl', 'models/table_tennis2_left.jpeg.pkl', 'models/table_tennis3.jpg.pkl', 'models/table_tennis3_left.jpg.pkl', 'models/table_tennis4.mp4.pkl', 'models/table_tennis5.mp4.pkl', 'models/table_tennis6.mp4.pkl'],
    }

def count_people(frame, pose):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return 1
    return 0

def main():
    words = ['archery', 'basketball', 'boxing', 'golf', 'Hurray', 'table_tennis']
    models = load_all_models()
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Checking the number of people. Please stay still for 7 seconds...")

    start_time = time.time()
    people_counts = []

    while time.time() - start_time < 7:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video device.")
            return

        frame = cv2.flip(frame, 1)
        people_counts.append(count_people(frame, pose))
        cv2.putText(frame, "Determine the number of people.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Charades', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    avg_people_count = round(sum(people_counts) / len(people_counts))
    print(f"Detected {avg_people_count} people.")

    if avg_people_count == 1:
        used_words = set()

        while len(used_words) < len(words):
            word = random.choice([w for w in words if w not in used_words])
            print(f"Perform the pose for: {word}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from video device.")
                    break

                frame = cv2.flip(frame, 1)
                cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Similarity: {0:.2f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Charades', frame)

                cv2.imwrite('current_frame.jpg', frame)
                user_pose = extract_pose_from_image('current_frame.jpg')

                if user_pose:
                    max_similarity = 0
                    for model_path in models[word]:
                        ref_pose = load_pose_model(model_path)
                        similarity = compare_poses(user_pose, ref_pose)
                        if similarity > max_similarity:
                            max_similarity = similarity

                    frame = cv2.flip(cap.read()[1], 1)
                    cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Similarity: {max_similarity:.2f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Charades', frame)

                    if max_similarity >= 0.7:
                        print("Correct Pose!")
                        cv2.putText(frame, 'Correct!', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Charades', frame)
                        cv2.waitKey(2000)
                        used_words.add(word)
                        break

                key = cv2.waitKey(1)
                if key & 0xFF == ord('n'):
                    used_words.add(word)
                    break
                elif key & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        print("All words have been used.")
        cap.release()
        cv2.destroyAllWindows()

    elif avg_people_count == 2:
        used_words = set()

        while len(used_words) < len(words):
            word = random.choice([w for w in words if w not in used_words])
            print(f"Both players perform the pose for: {word}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from video device.")
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                left_frame = frame[:, :width//2, :]
                right_frame = frame[:, width//2:, :]

                # 좌측 사용자 유사도 계산
                cv2.imwrite('left_frame.jpg', left_frame)
                left_pose = extract_pose_from_image('left_frame.jpg')
                left_similarity = 0
                if left_pose:
                    for model_path in models[word]:
                        ref_pose = load_pose_model(model_path)
                        similarity = compare_poses(left_pose, ref_pose)
                        if similarity > left_similarity:
                            left_similarity = similarity
                    cv2.putText(left_frame, f'Similarity: {left_similarity:.2f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                # 우측 사용자 유사도 계산
                cv2.imwrite('right_frame.jpg', right_frame)
                right_pose = extract_pose_from_image('right_frame.jpg')
                right_similarity = 0
                if right_pose:
                    for model_path in models[word]:
                        ref_pose = load_pose_model(model_path)
                        similarity = compare_poses(right_pose, ref_pose)
                        if similarity > right_similarity:
                            right_similarity = similarity
                    cv2.putText(right_frame, f'Similarity: {right_similarity:.2f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                # 프레임 합치기
                frame[:, :width//2, :] = left_frame
                frame[:, width//2:, :] = right_frame
                cv2.putText(frame, word, (width//2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Charades', frame)

                if left_similarity >= 0.7 or right_similarity >= 0.7:
                    if left_similarity >= 0.7:
                        print("Left player wins!")
                        cv2.putText(frame, 'Correct!', (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if right_similarity >= 0.7:
                        print("Right player wins!")
                        cv2.putText(frame, 'Correct!', (width//2 + 50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Charades', frame)
                    cv2.waitKey(2000)
                    used_words.add(word)
                    break

                key = cv2.waitKey(1)
                if key & 0xFF == ord('n'):
                    used_words.add(word)
                    break
                elif key & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    print("All words have been used.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
