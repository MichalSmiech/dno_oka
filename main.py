import io
import os
import numpy
import PySimpleGUI as sg
import skimage
from PIL import Image
from datetime import datetime
from simple_detector import SimpleDetector
from ai_detector import AiDetector

file_types = [("All files (*.*)", "*.*")]

def main():
    layout = [
        [
            sg.Column(
                [
                    [sg.Text("Obraz wejściowy")],
                    [sg.Image(key="-INPUT_IMG-")],
                ],
            ),
            sg.Column(
                [
                    [sg.Text("Maska ekspercja")],
                    [sg.Image(key="-MANUAL_IMG-")],
                ],
            )],
        [
            sg.Column(
                [
                    [sg.Text("Obraz wyjściowy")],
                    [sg.Image(key="-OUTPUT_IMG-")],
                ],
            ),
            sg.Column(
                [
                    [sg.Text("Macierz pomyłek")],
                    [sg.Image(key="-ERROR_IMG-")],
                ],
            ),
        ],
        [
            sg.Text("Image File"),
            sg.Input(size=(60, 1), key="-IMAGE_FILE-", default_text="data/images/02_h_800.jpg"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
        [
            sg.Text("Manual File"),
            sg.Input(size=(60, 1), key="-MANUAL_FILE-", default_text="data/manual/02_h_800.tif"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Manual"),
        ],
        [
            sg.Text("Mask File"),
            sg.Input(size=(60, 1), key="-MASK_FILE-", default_text="data/mask/02_h_mask_800.tif"),
            sg.FileBrowse(file_types=file_types),
        ],
        [
            sg.Button("Prosty klasyfikator"),
        ],
        [
            sg.Button("AI klasyfikator"),
        ],
        [
            sg.Text(key="-DATA-", text=""),
        ],
    ]

    window = sg.Window("Image Viewer", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-IMAGE_FILE-"]
            if os.path.exists(filename):
                image = Image.open(filename)
                image.thumbnail((500, 500))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-INPUT_IMG-"].update(data=bio.getvalue())
        if event == "Load Manual":
            filename = values["-MANUAL_FILE-"]
            if os.path.exists(filename):
                image = Image.open(filename)
                image.thumbnail((500, 500))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-MANUAL_IMG-"].update(data=bio.getvalue())
        if event == "Prosty klasyfikator":
            image_file_path = values["-IMAGE_FILE-"]
            manual_file_path = values["-MANUAL_FILE-"]
            mask_file_path = values["-MASK_FILE-"]
            detector = SimpleDetector()
            detector.load(image_file_path, manual_file_path, mask_file_path)
            detector.run()

            myarray = numpy.array(detector.result_img) * 255
            image = Image.fromarray(numpy.uint8(myarray))
            image.thumbnail((500, 500))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-OUTPUT_IMG-"].update(data=bio.getvalue())

            ppv, tpr, spc, avg, error_img = detector.stats()
            stats = f'trafność (FP): {ppv}'
            stats += f'\nczułość (FN): {tpr}'
            stats += f'\nswoistość (TN): {spc}'
            stats += f'\nśrednia arytmetyczna czułości i swoistości (TP): {avg}'
            window["-DATA-"].update(stats)

            skimage.io.imsave('error_img.jpg', error_img)
            image = Image.open('error_img.jpg')
            image.thumbnail((500, 500))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-ERROR_IMG-"].update(data=bio.getvalue())

        if event == "AI klasyfikator":
            image_file_path = values["-IMAGE_FILE-"]
            manual_file_path = values["-MANUAL_FILE-"]
            mask_file_path = values["-MASK_FILE-"]
            detector = AiDetector()
            detector.load(image_file_path, manual_file_path, mask_file_path)
            detector.run()

            myarray = numpy.array(detector.result_img) * 255
            image = Image.fromarray(numpy.uint8(myarray))
            image.thumbnail((500, 500))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-OUTPUT_IMG-"].update(data=bio.getvalue())

            ppv, tpr, spc, avg, error_img = detector.stats()
            stats = f'trafność (FP): {ppv}'
            stats += f'\nczułość (FN): {tpr}'
            stats += f'\nswoistość (TN): {spc}'
            stats += f'\nśrednia arytmetyczna czułości i swoistości (TP): {avg}'
            window["-DATA-"].update(stats)

            skimage.io.imsave('error_img.jpg', error_img)
            image = Image.open('error_img.jpg')
            image.thumbnail((500, 500))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-ERROR_IMG-"].update(data=bio.getvalue())


if __name__ == "__main__":
    main()