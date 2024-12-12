import cv2
import numpy as np

def main():
    # 템플릿 마스크 로드
    template_mask = cv2.imread('deeplab_red_edge_mask.jpg')
    if template_mask is None:
        raise ValueError("템플릿 이미지를 로드할 수 없습니다.")

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 템플릿 마스크를 더 크게 리사이징 (웹캠 화면의 80% 정도 크기로)
    template_height = int(480 * 0.8)  # 웹캠 높이의 80%
    template_mask = cv2.resize(template_mask, (640, template_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            # 프레임 크기 조정
            frame = cv2.resize(frame, (640, 480))
            
            # 결과 이미지 준비
            output_image = frame.copy()

            # 중앙 위치 계산
            start_y = (480 - template_height) // 2
            end_y = start_y + template_height

            # 템플릿 마스크를 중앙에 오버레이
            mask_area = output_image[start_y:end_y, :]
            
            # 마스크의 빨간색 부분만 오버레이 (투명도 조정)
            red_mask = template_mask[:, :, 2] > 0
            mask_area[red_mask] = mask_area[red_mask] * 0.3 + template_mask[red_mask] * 0.7

            # 결과 표시
            cv2.putText(output_image, "Press 'q' to quit", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Pose Template Overlay', output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()