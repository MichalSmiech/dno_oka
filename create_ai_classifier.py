from ai_detector import AiDetector

if __name__ == '__main__':
    detector = AiDetector()

    detector.create_classifier()
    detector.save_classifier()
