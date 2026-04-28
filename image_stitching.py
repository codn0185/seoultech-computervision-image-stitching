import cv2
import numpy as np


class ImageStitching:
    # === 샹수 ===

    WINDOW_TITLE: str = "Image Stitcher"  # 윈도우 제목
    RESULT_FILENAME: str = "Result.png"  # 저장할 파일명

    # === 멤버 변수 ===

    detector: cv2.Feature2D  # Feature Detector
    matcher: cv2.DescriptorMatcher  # Feature Matcher

    images: list = []  # 사용할 이미지 리스트
    result_image: np.ndarray  # 결과 이미지

    # 생성자
    def __init__(self):
        # Initialize Feature Detector
        self.detector = cv2.SIFT_create()  # SIFT
        # Initialize Feature Matcher
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    # 프로그램 실행
    def run(self, image_files: list[str] = []):
        self.images = self.load_image_files(image_files)
        if not self.images:
            raise Exception("Empty Image List")

        # 이미지 정합
        self.stitch_images()

        # 결과 저장 및 출력
        self.save_result()
        self.show_result()

    # 이미지 파일 불러오기
    def load_image_files(self, filenames: list[str]):
        imgs = []
        for filename in filenames:
            try:
                imgs.append(cv2.imread(filename))
            except:
                continue
        return imgs

    # 이미지 정합
    def stitch_images(self):
        pass

    # 결과 저장
    def save_result(self):
        cv2.imwrite(self.RESULT_FILENAME, self.result_image)

    # 결과 출력
    def show_result(self):
        cv2.imshow(self.WINDOW_TITLE, self.result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
