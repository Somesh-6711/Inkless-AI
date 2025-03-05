import cv2
from src.inkless_ai.components.air_writing import AirWriting

def main():
    """ Run real-time air-writing stroke recognition with gesture-based controls """
    cap = cv2.VideoCapture(1)  # Use 1 for external webcam, 0 for laptop camera
    air_writer = AirWriting()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = air_writer.process_frame(frame)  # Process frame
        cv2.imshow("Inkless AI - Air-Writing", frame)

        # Press 'Q' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
