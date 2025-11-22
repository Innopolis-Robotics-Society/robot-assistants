#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional

import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge


class AprilTagScaleNode(Node):
    """
    Нода:
      - подписывается на уже РЕКТИФИЦИРОВАННОЕ изображение (image_rect)
      - ищет чёрные квадраты (априль-теги как «чёрные коробки»)
      - меряет длину стороны каждого квадрата в пикселях
      - зная, что реальная сторона = 8 см (0.08 м), считает px/mm и px/m
      - публикует усреднённый px/m в топик hoba/pixels_per_meter
    """

    def __init__(self):
        super().__init__("apriltag_scale_node")

        # Параметры
        self.declare_parameter("image_topic", "/image_rect")
        self.declare_parameter("tag_size_m", 0.08)            # 8 см
        self.declare_parameter("min_area_px", 500.0)          # минимальная площадь квадрата
        self.declare_parameter("aspect_ratio_tol", 0.25)      # допустимое отклонение от квадрата
        self.declare_parameter("dark_thresh", 120.0)          # макс. средняя яркость внутри квадрата
        self.declare_parameter("debug_view", False)           # показывать окно с рисованием

        image_topic = self.get_parameter("image_topic").value
        self.tag_size_m = float(self.get_parameter("tag_size_m").value)
        self.min_area_px = float(self.get_parameter("min_area_px").value)
        self.aspect_ratio_tol = float(self.get_parameter("aspect_ratio_tol").value)
        self.dark_thresh = float(self.get_parameter("dark_thresh").value)
        self.debug_view = bool(self.get_parameter("debug_view").value)

        self.bridge = CvBridge()

        # Подписка на изображение (уже rect)
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        # Паблишер px/m
        self.px_per_m_pub = self.create_publisher(Float64, "hoba/pixels_per_meter", 10)

        self.get_logger().info(
            f"AprilTagScaleNode: subscribing to '{image_topic}', "
            f"tag_size_m={self.tag_size_m:.3f}, debug_view={self.debug_view}"
        )

    # ------------ основной callback ------------
    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Находим все кандидаты-квадраты и их длины сторон в пикселях
        side_lengths_px, debug_img = self.detect_black_squares(cv_img)

        if not side_lengths_px:
            # Ничего не нашли — просто игнорируем кадр
            self.get_logger().debug("No square candidates found")
            if self.debug_view and debug_img is not None:
                cv.imshow("apriltag_scale_debug", debug_img)
                cv.waitKey(1)
            return

        # Усредняем длину стороны квадрата в пикселях
        side_px_mean = float(np.mean(side_lengths_px))

        # Переводим в px/mm и px/m
        tag_size_mm = self.tag_size_m * 1000.0  # 0.08 м -> 80 мм
        px_per_mm = side_px_mean / tag_size_mm      # px/mm
        px_per_m = side_px_mean / self.tag_size_m   # px/m

        # Публикуем px/m
        msg_out = Float64()
        msg_out.data = px_per_m
        self.px_per_m_pub.publish(msg_out)

        self.get_logger().info(
            f"Detected {len(side_lengths_px)} squares | "
            f"side_px≈{side_px_mean:.2f} | px/mm≈{px_per_mm:.4f} | px/m≈{px_per_m:.2f}"
        )

        # Отладочное окно
        if self.debug_view and debug_img is not None:
            cv.putText(
                debug_img,
                f"side_px={side_px_mean:.1f}, px/m={px_per_m:.1f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            cv.imshow("apriltag_scale_debug", debug_img)
            cv.waitKey(1)

    # ------------ поиск чёрных квадратов ------------
    def detect_black_squares(self, img_bgr: np.ndarray) -> (List[float], Optional[np.ndarray]):
        """
        Ищет чёрные квадраты (априль-теги как «чёрные коробки»),
        возвращает список длины стороны (px) для каждого квадрата.
        """
        debug_img = img_bgr.copy()
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        # Немного сглаживаем, бинаризация по яркости с Otsu
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(
            blur,
            0,
            255,
            cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        # Морфологическое закрытие, чтобы объединить внутренние части тега
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)

        # Ищем контуры внешних объектов
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        side_lengths: List[float] = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_area_px:
                continue

            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            # Нужен выпуклый четырёхугольник
            if len(approx) != 4:
                continue
            if not cv.isContourConvex(approx):
                continue

            # Проверяем, что это примерно квадрат
            x, y, w, h = cv.boundingRect(approx)
            aspect = w / float(h)
            if not (1.0 - self.aspect_ratio_tol <= aspect <= 1.0 + self.aspect_ratio_tol):
                continue

            # Проверяем, что внутри темно (чёрный квадрат)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv.drawContours(mask, [approx], -1, 255, -1)
            mean_val = cv.mean(gray, mask=mask)[0]
            if mean_val > self.dark_thresh:
                # Слишком светлый, не принимаем
                continue

            # Считаем длину стороны по четырём сторонам многоугольника
            pts = approx.reshape(-1, 2)
            lengths = []
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                d = math.hypot(float(p1[0] - p2[0]), float(p1[1] - p2[1]))
                lengths.append(d)
            side_px = float(np.mean(lengths))
            side_lengths.append(side_px)

            # Для отладки — рисуем контур и подписываем длину
            cv.polylines(debug_img, [approx], True, (0, 255, 0), 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv.putText(
                debug_img,
                f"{side_px:.1f}px",
                (cx - 30, cy),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        return side_lengths, debug_img


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagScaleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Закрыть окно, если debug_view включён
        try:
            cv.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
