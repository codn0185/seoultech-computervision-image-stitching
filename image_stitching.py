import cv2
import numpy as np
import glob


class ImageStitching:

    # === 샹수 ===
    SOURCE_IMAGE_DIRECTORY: str = "./images"  # 입력 이미지 디렉토리
    IMAGE_FILE_EXTENSIONS: tuple[str] = ("jpg", "jpeg", "png")  # 이미지 파일 확장자
    RESULT_FILENAME: str = "Result.png"  # 저장할 파일명

    WINDOW_TITLE: str = "Image Stitcher"  # 윈도우 제목

    # === 멤버 변수 ===

    detector: cv2.Feature2D  # Feature Detector
    matcher: cv2.DescriptorMatcher  # Feature Matcher

    source_images: list[np.ndarray] = []  # 사용할 이미지 리스트
    result_image: np.ndarray  # 결과 이미지

    # 생성자
    def __init__(self):
        # Initialize Feature Detector
        self.detector = cv2.SIFT_create()  # SIFT
        # Initialize Feature Matcher
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    # 프로그램 실행
    def run(self):
        self.initialize_source_images()
        if not self.source_images:
            raise Exception("Empty Image List")

        # 이미지 정합
        self.stitch_images()

        # 결과 저장 및 출력
        self.save_result()
        self.show_result()

    def initialize_source_images(self):
        image_paths = []
        for ext in self.IMAGE_FILE_EXTENSIONS:
            image_paths.extend(glob.glob(f"{self.SOURCE_IMAGE_DIRECTORY}/*.{ext}"))
        self.source_images = [cv2.imread(path) for path in image_paths]

    # 이미지 정합
    def stitch_images(self):
        self.result_image = self.source_images[len(self.source_images) // 2]  # 중앙 이미지 기준으로 병합

        for i in range(0, len(self.source_images)):
            # 정합할 두 이미지 선택
            img1 = self.result_image
            img2 = self.source_images[i]

            # key points & descriptors 얻기
            kp1, desc1 = self.detector.detectAndCompute(img1, None)
            kp2, desc2 = self.detector.detectAndCompute(img2, None)

            # 매칭
            matches = self.matcher.match(desc2, desc1)
            if len(matches) < 4:
                print(f"Not enough matches are found at image index {i} - {len(matches)} < 4")
                break
            src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Homography 계산
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # img2를 img1에 맞게 변환
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            result_corners = np.concatenate(
                (
                    np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
                    dst,
                ),
                axis=0,
            )
            [xmin, ymin] = np.int32(result_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(result_corners.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]
            H_translate = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            # 이미지 워핑 및 병합
            result_warped = cv2.warpPerspective(img2, H_translate @ H, (xmax - xmin, ymax - ymin))
            # img1을 같은 크기의 캔버스에 위치시킴
            img1_canvas = np.zeros_like(result_warped)
            img1_canvas[t[1] : h1 + t[1], t[0] : w1 + t[0]] = img1

            # Feather blending 메서드 사용
            self.result_image = self.feather_blend(img1_canvas, result_warped)

    # 두 이미지 블렌딩
    def feather_blend(self, img1_canvas: np.ndarray, result_warped: np.ndarray) -> np.ndarray:
        # 마스크 생성
        mask1 = (cv2.cvtColor(img1_canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        mask2 = (cv2.cvtColor(result_warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # distance transform (float32)
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)

        # 가중치 계산 (겹치는 부분은 두 distance의 합으로 정규화)
        weight1 = dist1 / (dist1 + dist2 + 1e-8)
        weight2 = dist2 / (dist1 + dist2 + 1e-8)
        weight1 = np.nan_to_num(weight1)
        weight2 = np.nan_to_num(weight2)

        # 3채널로 확장
        weight1_3c = np.repeat(weight1[:, :, np.newaxis], 3, axis=2)
        weight2_3c = np.repeat(weight2[:, :, np.newaxis], 3, axis=2)

        # feather blending
        blended = (img1_canvas.astype(np.float32) * weight1_3c + result_warped.astype(np.float32) * weight2_3c).astype(
            np.uint8
        )

        return blended

    # 결과 저장
    def save_result(self):
        cv2.imwrite(self.RESULT_FILENAME, self.result_image)

    # 결과 출력
    def show_result(self):
        cv2.imshow(self.WINDOW_TITLE, self.result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
