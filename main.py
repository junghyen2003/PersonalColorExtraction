from hair_segmentation import FaceHairSegmentation
import cv2

def process_image(image_path):
    # 인스턴스 생성
    segmenter = FaceHairSegmentation()

    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 감지
    faces = segmenter.detect_faces(image_rgb)

    # 얼굴별로 처리
    for face in faces:
        image = segmenter.process_face(image, face)

    return image

if __name__ == "__main__":
    image_path = "images/test4.jpeg"

    # 이미지 처리
    processed_image = process_image(image_path)

    # 결과 이미지 출력
    cv2.imshow('Face Mesh with Expanded Hair Segmentation', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()