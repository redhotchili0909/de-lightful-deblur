import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter
from datetime import datetime
import os

from streak_detect import detect

class StreakDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streak Detection")
        self.setGeometry(100, 100, 800, 600)

        # Load Image Button
        self.load_button = QPushButton("Load Image", self)
        self.load_button.setGeometry(10, 10, 100, 30)
        self.load_button.clicked.connect(self.load_image)

        # Run Detection Button
        self.run_button = QPushButton("Run Streak Detection", self)
        self.run_button.setGeometry(120, 10, 150, 30)
        self.run_button.clicked.connect(self.run_streak_detection)
        self.run_button.setEnabled(False)

        # Go Back to Original Button
        self.reset_button = QPushButton("Go Back to Selection", self)
        self.reset_button.setGeometry(280, 10, 150, 30)
        self.reset_button.clicked.connect(self.reset_to_original)
        self.reset_button.setEnabled(False)

        # Message Label
        self.message_label = QLabel(self)
        self.message_label.setGeometry(10, 45, 780, 20)
        self.message_label.setText("")

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 70, 780, 520)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.start_selection
        self.image_label.mouseMoveEvent = self.update_selection
        self.image_label.mouseReleaseEvent = self.end_selection

        # Initialize variables
        self.image = None
        self.original_image = None
        self.selection = None
        self.selection_rect = None
        self.start_point = None
        self.drawing = False
        self.detection_ran = False

    def load_image(self):
        # Load image and display
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_image = self.image.copy()
            self.display_image(self.image)
            self.run_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.detection_ran = False


    def display_image(self, image):
        # Display the given image in the QLabel
        if len(image.shape) == 2:
            # Grayscale image
            height, width = image.shape
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:
            # Color image
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def reset_to_original(self):
        # Restore the original image in the QLabel
        self.image = self.original_image.copy()
        self.display_image(self.image)
        self.selection = None
        self.reset_button.setEnabled(False)

    def start_selection(self, event):
        if self.image is None:
            return
        # Start drawing the selection rectangle
        self.start_point = event.pos()
        self.selection_rect = QRect(self.start_point, self.start_point)
        self.drawing = True

    def update_selection(self, event):
        # Update rectangle only if drawing is active
        if self.drawing:
            self.selection_rect = QRect(self.start_point, event.pos()).normalized()
            self.update_display()

    def end_selection(self, event):
        # End drawing mode on mouse release
        if self.drawing:
            self.drawing = False
            self.selection = self.selection_rect
            self.update_display()
            print("Selection:", self.selection)

    def update_display(self):
        # Draw selection rectangle on the displayed image
        if self.image is not None:
            if len(self.image.shape) == 2:
                # Grayscale image
                height, width = self.image.shape
                q_image = QImage(self.image.data, width, height, width, QImage.Format_Grayscale8)
            else:
                # Color image
                height, width, channel = self.image.shape
                bytesPerLine = 3 * width
                image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            painter = QPainter(pixmap)
            if self.selection_rect:
                painter.setPen(Qt.red)
                painter.drawRect(self.selection_rect)
            painter.end()
            self.image_label.setPixmap(pixmap)

    def run_streak_detection(self):
        # Ensure selection is made
        if self.selection and self.image is not None:
            x1, y1, x2, y2 = self.selection.getCoords()
            selected_region = self.image[y1:y2, x1:x2]

            # Create an output directory with a timestamp if detection runs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_subdir = os.path.join("manual_select", f"streak_detection_{timestamp}")
            os.makedirs(output_subdir, exist_ok=True)

            # Run streak detection on the selected region
            output_image, contours = detect(selected_region, output_subdir)

            # Update the selected region in the full image and display
            self.image[y1:y2, x1:x2] = output_image
            self.display_image(self.image)

            # Enable the reset button and mark detection as run
            self.reset_button.setEnabled(True)
            self.detection_ran = True

            # Save the updated full image if detection was successful
            full_image_filename = os.path.join(output_subdir, f"processed_full_image_{timestamp}.png")
            cv2.imwrite(full_image_filename, self.image)
            print(f"Streak detection results saved as {full_image_filename}")

            # Update the message label
            self.message_label.setText(f"Image saved: {full_image_filename}")
            QTimer.singleShot(5000, lambda: self.message_label.setText(""))
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = StreakDetectionGUI()
    main_window.show()
    sys.exit(app.exec_())
