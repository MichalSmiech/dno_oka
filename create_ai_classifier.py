from simple_detector import SimpleDetector
from ai_detector import AiDetector
from datetime import datetime

def ai():
    detector = AiDetector()

    detector.create_classifier()
    detector.save_classifier()

if __name__ == '__main__':
    ai()