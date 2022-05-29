from simple_detector import SimpleDetector
from ai_detector import AiDetector
from datetime import datetime


def simple():
    detector = SimpleDetector()

    name = '02_h'
    detector.load(f'data/images/{name}.jpg', f'data/manual/{name}.tif', f'data/mask/{name}_mask.tif')

    detector.filter()

    detector.compare_with_manual()

    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def ai():
    start = datetime.now().timestamp()
    detector = AiDetector()

    # detector.load_classifier()
    detector.create_classifier()
    detector.save_classifier()
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 10000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    # detector.predict_image()
    # print(start - datetime.now().timestamp())



if __name__ == '__main__':
    ai()