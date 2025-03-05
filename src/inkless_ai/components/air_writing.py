import time
from src.inkless_ai.components.handwriting_recognition import recognize_handwriting

class AirWriting:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        """
        Initializes hand tracking for air-writing with gesture-based controls.
        """
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.strokes = []  # Stores strokes for air-writing
        self.current_stroke = []  # Stores points for the current character
        self.drawing = False  # Indicates whether writing is active
        self.word_detected = False  # Indicates if a word has been detected
        self.last_detection_time = 0  # Timer for restarting drawing

    def process_frame(self, frame):
        """
        Detects hand and tracks index fingertip movement.
        :param frame: Input frame from webcam
        :return: Processed frame with air-writing strokes
        """
        frame = cv2.flip(frame, 1)  # Fix mirroring issue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if self.word_detected:
            # Stop writing for 2-4 seconds after word detection
            if time.time() - self.last_detection_time > 3:  # Restart after 3 seconds
                self.word_detected = False
            return frame

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                x, y = int(index_finger.x * w), int(index_finger.y * h)

                if self.is_palm_open(hand_landmarks):
                    self.strokes = []
                    self.current_stroke = []
                    return frame

                if self.is_only_index_finger_up(hand_landmarks):
                    self.drawing = True
                else:
                    self.drawing = False

                if self.drawing:
                    self.current_stroke.append((x, y))

                for stroke in self.strokes:
                    for i in range(1, len(stroke)):
                        cv2.line(frame, stroke[i - 1], stroke[i], (0, 255, 0), 2)

                for i in range(1, len(self.current_stroke)):
                    cv2.line(frame, self.current_stroke[i - 1], self.current_stroke[i], (255, 0, 0), 2)

        return frame

    def stop_drawing_temporarily(self):
        """
        Stops drawing after word detection and restarts after a delay.
        """
        recognized_text = recognize_handwriting(self.strokes)
        print(f"Recognized Word: {recognized_text}")

        self.word_detected = True
        self.last_detection_time = time.time()
        self.strokes.append(self.current_stroke)
        self.current_stroke = []
