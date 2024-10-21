from sklearn.cluster import KMeans
import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceHairSegmentation:
    def __init__(self, tflite_model_path='model/hair_segmenter.tflite', min_face_size=10):
        # MTCNN 초기화
        self.detector = MTCNN(min_face_size=min_face_size)

        # Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Haar Cascade 초기화
        self.eyes_cascade = cv2.CascadeClassifier('model/haarcascade_eye.xml')

        # TensorFlow GPU 확인
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        # MediaPipe ImageSegmenter 초기화
        base_options = python.BaseOptions(model_asset_path=tflite_model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_confidence_masks=True,
                                               output_category_mask=False)
        self.hair_segmenter = vision.ImageSegmenter.create_from_options(options)

    def detect_faces(self, image_rgb):
        """MTCNN을 사용하여 얼굴을 감지합니다."""
        return self.detector.detect_faces(image_rgb)

    def apply_face_mesh(self, face_image_rgb):
        """MediaPipe Face Mesh를 사용하여 얼굴 메쉬를 처리합니다."""
        with self.mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            return face_mesh.process(face_image_rgb)

    def segment_hair(self, expanded_face_image_rgb):
        """머리카락 영역을 세그멘테이션합니다."""
        hair_segmentation_result = self.hair_segmenter.segment(mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.array(expanded_face_image_rgb)))

        return hair_segmentation_result.confidence_masks

    def draw_face_mesh(self, face_image, face_landmarks):
        """얼굴 메쉬를 그립니다."""
        self.mp_drawing.draw_landmarks(face_image, face_landmarks,
                                       self.mp_face_mesh.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                                         circle_radius=1),
                                       connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                           thickness=1,
                                                                                           circle_radius=1))

    def extract_hair_color(self, expanded_face_image, hair_mask):
        """머리카락 영역을 추출하여 해당 색상을 반환합니다."""
        hair_label_index = 1  # 머리카락 라벨
        hair_color = (0, 0, 0)  # 기본값 초기화

        if hair_mask is not None and len(hair_mask) > hair_label_index:
            hair_mask_np = hair_mask[hair_label_index].numpy_view()

            # 머리카락 마스크를 이진화 처리
            hair_mask_np = (hair_mask_np > 0.5).astype(np.uint8)

            # 머리카락 색상 추출
            hair_color_bgr = cv2.mean(expanded_face_image, mask=hair_mask_np.astype(np.uint8))[:3]

            # BGR -> RGB 변환
            hair_color_rgb = (hair_color_bgr[2], hair_color_bgr[1], hair_color_bgr[0])  # BGR -> RGB

            return hair_color_rgb  # 머리카락 색상 반환
        else:
            return hair_color  # 기본값 반환

    def extract_skin_color(self, face_image, face_landmarks):
        """Face Mesh 랜드마크를 기반으로 피부 색상을 추출하고 메쉬를 그립니다."""
        # 랜드마크의 인덱스 정의
        landmark_indices = [1, 2, 5, 6, 10, 11, 12, 13]  # 필요에 따라 조정 가능

        # 랜드마크의 좌표를 기반으로 피부 영역을 정의
        points = []
        for idx in landmark_indices:
            point = face_landmarks.landmark[idx]
            x = int(point.x * face_image.shape[1])  # 이미지의 너비에 비례하여 x좌표 계산
            y = int(point.y * face_image.shape[0])  # 이미지의 높이에 비례하여 y좌표 계산
            points.append((x, y))

        # 피부 영역의 다각형 생성
        skin_mask = np.zeros(face_image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(skin_mask, np.array(points, dtype=np.int32), 1)

        # 피부 영역 추출
        skin_area = face_image * skin_mask[:, :, np.newaxis]  # 피부 영역만 남김

        # 피부 색상 추출 (BGR)
        skin_color_bgr = cv2.mean(skin_area, mask=skin_mask)[:3]

        # BGR -> RGB 변환
        skin_color_rgb = (skin_color_bgr[2], skin_color_bgr[1], skin_color_bgr[0])

        return skin_color_rgb  # 피부 색상 및 메쉬가 그려진 이미지 반환

    def display_color_info_on_image(self, image, hair_color, skin_color, eye_color, position):
        """머리카락, 피부, 눈 색상 정보를 본 이미지에 시각적으로 표시합니다."""
        x, y = position
        color_box_size = 30  # 색상 박스의 크기

        # 머리카락 색상 표시
        cv2.rectangle(image, (x, y), (x + color_box_size, y + color_box_size), hair_color[::-1], -1)  # BGR로 변환
        cv2.putText(image, 'Hair', (x, y + color_box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 피부 색상 표시
        cv2.rectangle(image, (x + color_box_size, y), (x + 2 * color_box_size, y + color_box_size),
                      skin_color[::-1], -1)
        cv2.putText(image, 'Skin', (x + color_box_size, y + color_box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # 눈 색상 표시
        cv2.rectangle(image, (x + 2 * color_box_size, y), (x + 3 * color_box_size, y + color_box_size),
                      eye_color[::-1], -1)
        cv2.putText(image, 'Eye', (x + 2 * color_box_size, y + color_box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    def get_hair_skin_eye_colors(self, face_image, expanded_face_image, hair_mask, eye_region, results):
        """머리카락, 피부, 눈 색상을 분석합니다."""
        hair_color = self.extract_hair_color(expanded_face_image, hair_mask)  # 머리카락 색상 추출
        # 피부 색상 추출 (Face Mesh 랜드마크 포함)
        skin_color = self.extract_skin_color(face_image, results.multi_face_landmarks[0])

        # 눈동자 색상 추출
        if eye_region is not None:
            eye_color = cv2.mean(eye_region)[:3]  # BGR로 변환
            eye_color_rgb = (eye_color[2], eye_color[1], eye_color[0])  # BGR -> RGB
        else:
            eye_color_rgb = (0, 0, 0)  # 기본값

        return hair_color, skin_color, eye_color_rgb  # 3개 값 반환

    def expand_face_area(self, image, x, y, width, height, margin_ratio=3.0):
        """얼굴 영역을 확장합니다."""
        x_min = max(int(x - width * margin_ratio), 0)
        y_min = max(int(y - height * margin_ratio), 0)
        x_max = min(int(x + width * (1 + margin_ratio)), image.shape[1])
        y_max = min(int(y + height * (1 + margin_ratio)), image.shape[0])

        # 확장된 얼굴 이미지의 크기가 0이 아닌지 확인
        if x_max > x_min and y_max > y_min:
            expanded_face_image = image[y_min:y_max, x_min:x_max]
            return expanded_face_image, (x_min, y_min, x_max, y_max)
        else:
            return None, (0, 0, 0, 0)  # 잘못된 경우

    def process_face(self, image, face):
        """단일 얼굴을 처리하고 머리카락, 피부, 눈 색상을 분석합니다."""
        x, y, width, height = face['box']
        face_image = image[y:y + height, x:x + width]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Haar Cascade를 사용하여 눈 감지
        gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = self.eyes_cascade.detectMultiScale(gray_face_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        eye_region = None
        for (ex, ey, ew, eh) in eyes:
            # 눈 영역의 중심 계산
            eye_center_x = ex + ew // 2
            eye_center_y = ey + eh // 2

            # 새 눈 영역의 크기를 기존 크기의 5%로 설정
            eye_width = int(ew * 0.1)  # 크기를 10%로 줄임
            eye_height = int(eh * 0.1)  # 크기를 10%로 줄임

            # 눈 영역의 좌표 조정 (정중앙을 기준으로)
            ex_new = max(eye_center_x - eye_width // 2, 0)
            ey_new = max(eye_center_y - eye_height // 2, 0)

            # 눈 영역 추출
            eye_region = face_image[ey_new:ey_new + eye_height, ex_new:ex_new + eye_width]

        # Face Mesh 처리
        results = self.apply_face_mesh(face_image_rgb)

        # 얼굴 주변 확장
        expanded_face_image, (x_min, y_min, x_max, y_max) = self.expand_face_area(image, x, y, width, height)

        # 확장된 얼굴 이미지가 None이 아닌지 확인
        if expanded_face_image is not None:
            # 머리카락 세그멘테이션 처리
            hair_mask = self.segment_hair(expanded_face_image)

            # 머리카락, 피부, 눈 색상 추출
            hair_color, skin_color, eye_color = self.get_hair_skin_eye_colors(face_image,
                                                                              expanded_face_image, hair_mask,
                                                                              eye_region, results)

            # 색상 정보를 이미지에 표시
            self.display_color_info_on_image(image, hair_color, skin_color, eye_color, (x_min, y_min))
        else:
            print("Expanded face area is None, skipping face processing.")

        return image
