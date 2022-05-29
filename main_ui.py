import io
import os
import numpy
import PySimpleGUI as sg
from PIL import Image
from datetime import datetime

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

def main():
    layout = [
        [sg.Column(
            [
                [sg.Text("Obraz wejściowy")],
                [sg.Image(key="-INPUT_IMG-")],
            ],
        ),
            sg.Column(
                [
                    [sg.Text("Obraz wyjściowy")],
                    [sg.Image(key="-OUTPUT_IMG-")],
                ],
            )],
        [
            sg.Text("Image File"),
            sg.Input(size=(60, 1), key="-IMAGE_FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
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
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-INPUT_IMG-"].update(data=bio.getvalue())
                window["-DATA-"].update('bio.getvalue()')
        if event == "Prosty klasyfikator":
            pass
        if event == "AI klasyfikator":
            pass


if __name__ == "__main__":
    main()