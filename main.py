import random
import cv2
from pose_comparator import compare_poses, load_pose_model
from pose_extractor import extract_pose_from_image

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
        'archery': ['models/archery1.jpg.pkl', 'models/archery2.jpg.pkl', 'models/archery3.jpg.pkl', 'models/archery4.mp4.pkl'],
        'Hurray': ['models/Hurray1.jpeg.pkl', 'models/Hurray2.jpeg.pkl'],
        'matrix': ['models/matrix1.jpeg.pkl', 'models/matrix2.jpg.pkl', 'models/matrix3.mp4.pkl']
    }

def main():
    words = ['archery', 'Hurray', 'matrix']
    models = load_all_models()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        word = random.choice(words)
        print(f"Perform the pose for: {word}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video device.")
                break

            cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Pose Game', frame)

            cv2.imwrite('current_frame.jpg', frame)
            user_pose = extract_pose_from_image('current_frame.jpg')

            if user_pose:
                max_similarity = 0
                for model_path in models[word]:
                    ref_pose = load_pose_model(model_path)
                    similarity = compare_poses(user_pose, ref_pose)
                    if similarity > max_similarity:
                        max_similarity = similarity

                cv2.putText(frame, f'Similarity: {max_similarity:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Pose Game', frame)

                if max_similarity >= 0.7:
                    print("Correct Pose!")
                    cv2.putText(frame, 'Correct!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Pose Game', frame)
                    cv2.waitKey(2000)
                    break

            key = cv2.waitKey(1)
            if key & 0xFF == ord('n'):  # 'n' 키를 누르면 다음 문제로 넘어감
                break
            elif key & 0xFF == ord('q'):  # 'q' 키를 누르면 프로그램 종료
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
