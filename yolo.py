import cv2 as cv
import torch
from ultralytics import YOLO


class YOLOCamera:
    """
    Класс для захвата видео с камеры и обработки кадров моделью YOLO.
    """

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 1280,
        height: int = 720,
        conf_threshold: float = 0.25,
        warmup_frames: int = 45,
    ):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.conf_threshold = conf_threshold
        self.warmup_frames = warmup_frames

        self.cap = None
        self.model = None

    # ------------------------------------------------------------------
    def init_camera(self):
        """Инициализация камеры и установка параметров."""
        self.cap = cv.VideoCapture(self.cam_index)

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

        # Warm-up (стабилизация автоэкспозиции и автоББ)
        for _ in range(self.warmup_frames):
            self.cap.read()
            cv.waitKey(1)

        # Настройки экспозиции/ББ (если поддерживаются драйвером)
        self.cap.set(cv.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv.CAP_PROP_WB_TEMPERATURE, 4600)

        # OpenCV → V4L2: AUTO_EXPOSURE = 0.25 == manual
        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv.CAP_PROP_GAIN, 0)

    # ------------------------------------------------------------------
    def init_model(self, weights: str = "yolov8s.pt"):
        """Загрузка модели YOLO."""
        self.model = YOLO(weights)

        if torch.cuda.is_available():
            self.model.to("cuda")

    # ------------------------------------------------------------------
    def run(self):
        """
        Основной цикл обработки видео.
        Нажмите ESC или 'q' для выхода.
        """
        if self.cap is None:
            self.init_camera()

        if self.model is None:
            self.init_model()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("❌ Ошибка чтения кадра.")
                break

            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            annotated = results.plot()

            cv.imshow("YOLO Camera", annotated)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        self.close()

    # ------------------------------------------------------------------
    def close(self):
        """Освобождение ресурсов."""
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cam = YOLOCamera(cam_index=0, width=1280, height=720)
    cam.run()
