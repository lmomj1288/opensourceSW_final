import pickle
from scipy.spatial import distance

def load_pose_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def compare_poses(user_pose, reference_pose):
    distances = [distance.euclidean((lm["x"], lm["y"], lm["z"]), (ref["x"], ref["y"], ref["z"])) for lm, ref in zip(user_pose, reference_pose)]
    similarity = 1 - (sum(distances) / len(distances))  # 유사도 계산
    return similarity

if __name__ == "__main__":
    user_pose = load_pose_model('output/user_pose.pkl')
    reference_pose = load_pose_model('models/pose1_model.pkl')
    similarity = compare_poses(user_pose, reference_pose)
    print(f"Pose similarity: {similarity:.2f}")
