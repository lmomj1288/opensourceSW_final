import pickle
from scipy.spatial import distance

def load_pose_model(model_path):
    """저장된 포즈 모델 로드"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def compare_poses(user_pose, reference_pose):
    """포즈 유사도 계산 (유클리드 거리)"""
    distances = [distance.euclidean((lm["x"], lm["y"], lm["z"]), (ref["x"], ref["y"], ref["z"])) for lm, ref in zip(user_pose, reference_pose)]
    similarity = 1 - (sum(distances) / len(distances))  # 유사도 계산
    return similarity

if __name__ == "__main__":
    user_pose = load_pose_model('output/user_pose.pkl')
    reference_pose = load_pose_model('models/pose1_model.pkl')
    similarity = compare_poses(user_pose, reference_pose)
    print(f"Pose similarity: {similarity:.3f}")
