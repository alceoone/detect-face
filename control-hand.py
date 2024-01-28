import cv2
import numpy as np
import subprocess

# Fungsi untuk menggeser aplikasi
def move_app(x, y):
    pass  # Ganti dengan implementasi sesuai kebutuhan di sistem operasi Mac

# Fungsi untuk menutup aplikasi
def close_app():
    subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "q" using command down'])

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve accuracy
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Apply thresholding to segment the hand
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (hand)
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)

            # Get the centroid of the hand
            M = cv2.moments(hand_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Move the application window if the centroid is above the center
                if cy < frame.shape[0] // 2:
                    move_app(cx, cy)

                # Close the application if the centroid is below the center
                else:
                    close_app()

        # Draw the contours on the original frame
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 3)

        # Display the frame
        cv2.imshow('Hand Tracking App Control', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
