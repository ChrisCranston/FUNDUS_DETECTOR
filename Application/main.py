from io import BytesIO
from os import getcwd, path
import PySimpleGUI as sg
from PIL import Image
from detector.detect import run_detector
from classifier.classify import fuzzy_classifier
# ---------------------------------------------
# main.py - 
# File controlling the GUI creation and events for application
#
# @Author - Chris Cranston W18018468
# ---------------------------------------------

# set theme of GUI
sg.theme('DarkBlue')
# set file types for "patient file"
file_types = [("TXT Files", "*.txt"),
              ("All files (*.*)", "*.*")]


def main():
    # set global variables
    inference_showing = False
    application_path = getcwd()
    input_folder = "/detector/input/test"
    target = application_path + input_folder
    patient_info = ""

    # Arrange layout for GUI
    layout = [
        [
            sg.Push(),
            sg.Text("FUNDUS Diabetes Marker Detection Application v1.0",
                    font=('Any 20')),
            sg.Push(),
        ],
        [

            sg.Text("Load Patient Information", pad=5),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types, initial_folder=target, tooltip="Browse for a patient file, currently using a .txt file"),
            sg.Button("Load Images", tooltip="Load images from the path and display"),
            sg.Button("Reset App", tooltip="Reset app to default layout"),
        ],
        [
            sg.Image(size=(512, 512), background_color="white", key="-IMAGE0-"),
            sg.Image(size=(512, 512), background_color="white", key="-IMAGE1-"),
        ],
        [
            sg.Text("Image Left:", key="-IMAGE_LEFT_TEXT-", expand_x=False),
            sg.Text("Image Right:", key="-IMAGE_RIGHT_TEXT-", expand_x=False)
        ],
        [ 
            sg.Checkbox('Show Confidence Values', default=True, key="-SHOWCONF-", tooltip="If selected, include confidence values when perfoming inferencing."),            
        ],
        [ 
            sg.Text("Confidence Threshold:"),
            sg.Slider(range=(0, 1), default_value=0.2, size=(50, 10), orientation="h",
                enable_events=True, resolution=0.05, key="-CONF_SLIDER-", tooltip="Control the confidence threshold when performing inferencing.")
        ],
        [
            sg.Button("Perform Inferencing", disabled=True, tooltip="Perform inferencing and show inferenced images"),
            sg.VSeparator(),
            sg.Button("Show/Hide Results", disabled=True, tooltip="Switch between images with and without labelling."),
        ],
        [
            sg.Multiline(key="patient_info", autoscroll=False,
                         size=(45, 3), no_scrollbar=True),
        ],
        [
            sg.Text("NDESP Grade:",key="-NDESP-", font=('Any 20')),
            sg.Text("", key="-RESULT-", font=('Any 20')),
        ],
        [ 
            sg.Text("Recommended action:", key="-REC-"),
        ],
    ]

    window = sg.Window("FUNDUS Diabetes Marker Detection Application",
                       layout, element_justification='center')
    # initiate even listener loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # parse path, load images and store in global array
        if event == "Load Images":
            patient_info = ""
            plain_images = []
            inferenced_images = []
            text_file = values["-FILE-"]
            folder = text_file.rsplit('/', 1)[0]
            if path.exists(text_file):
                temp = open(text_file, 'r').read().splitlines()
                for x, line in enumerate(temp):
                    if (x < 2):
                        print(plain_images)
                        image_path = folder+"/"+line
                        image = Image.open(image_path)
                        image.thumbnail((512, 512))
                        bio = BytesIO()
                        image.save(bio, format="PNG")
                        plain_images.append(image_path)
                        #update images and file location strings
                        if x == 0:
                            window['-IMAGE_LEFT_TEXT-'].update(image_path)
                        else:
                            window['-IMAGE_RIGHT_TEXT-'].update(image_path)
                        target_window = "-IMAGE"+str(x)+"-"
                        window[target_window].update(data=bio.getvalue(),visible=True)
                    else:
                        patient_info += line + "\n"
                window["patient_info"].update(patient_info)
                window["Perform Inferencing"].update(disabled=False)
                window["Show/Hide Results"].update(disabled=False)

        # Parse confidence threshold, display confidence and globally stored files and perform inferencing with yolov5
        if event == "Perform Inferencing":
            if (folder):
                inferenced_images = []
                inferenced = []
                if values["-SHOWCONF-"] == False:
                    inferenced, haemorrhage_count, exudate_count = run_detector(folder, False, values['-CONF_SLIDER-'])
                else:
                    inferenced, haemorrhage_count, exudate_count = run_detector(folder, True, values['-CONF_SLIDER-'])
                for x, result_image in enumerate(inferenced):
                    if (x < 2):
                        image = Image.open(result_image)
                        image.thumbnail((512, 512))
                        bio = BytesIO()
                        image.save(bio, format="PNG")
                        #update images and file location strings
                        if x == 0:
                            window['-IMAGE_RIGHT_TEXT-'].update(result_image)
                            inferenced_images.append(result_image)
                            target_window = "-IMAGE1-"
                        else:
                            window['-IMAGE_LEFT_TEXT-'].update(result_image)
                            inferenced_images.append(result_image)
                            target_window = "-IMAGE0-"
                        
                        window[target_window].update(data=bio.getvalue())
                        inference_showing = True

                # Perform fuzzy inferencing with results of yolov5
                fuz_result = fuzzy_classifier(haemorrhage_count,exudate_count)
                if (fuz_result):
                    if (fuz_result <= 33):
                        window['-RESULT-'].update("R0", text_color="green")
                        window['-NDESP-'].update(text_color="green")
                        window['-REC-'].update("Recommended action: Continue with regular screening period.", text_color="green")
                    elif (fuz_result <= 66):
                        window['-RESULT-'].update("R1", text_color="orange")
                        window['-NDESP-'].update(text_color="orange")
                        window['-REC-'].update("Recommended action: Increase screening regularity.", text_color="orange")
                    else:
                        window['-RESULT-'].update("R2", text_color="red")
                        window['-NDESP-'].update(text_color="red")
                        window['-REC-'].update("Recommended action: Referall to hospital.", text_color="red")

        # Show inference results on / off switch
        if event == "Show/Hide Results":
            if (inference_showing == True):
                for x, result_image in enumerate(plain_images):
                    if (x < 2):
                        image = Image.open(result_image)
                        image.thumbnail((512, 512))
                        bio = BytesIO()
                        image.save(bio, format="PNG")
                        if x == 0:
                            window['-IMAGE_LEFT_TEXT-'].update(result_image)
                        else:
                            window['-IMAGE_RIGHT_TEXT-'].update(result_image)
                        target_window = "-IMAGE"+str(x)+"-"
                        window[target_window].update(data=bio.getvalue())
                        inference_showing = False
            else:
                for x, result_image in enumerate(inferenced_images):
                    if (x < 2):
                        image = Image.open(result_image)
                        image.thumbnail((512, 512))
                        bio = BytesIO()
                        image.save(bio, format="PNG")
                        if x == 0:
                            window['-IMAGE_RIGHT_TEXT-'].update(result_image)
                            target_window = "-IMAGE1-"
                        else:
                            window['-IMAGE_LEFT_TEXT-'].update(result_image)
                            target_window = "-IMAGE0-"
                        window[target_window].update(data=bio.getvalue())
                        inference_showing = True

        # Reset variables 
        if event == "Reset App":
            window['-RESULT-'].update("", text_color="white")
            window['-NDESP-'].update(text_color="white")
            window['-REC-'].update("Recommended action:",text_color="white")
            window["patient_info"].update("")
            window["Perform Inferencing"].update(disabled=True)
            window["Show/Hide Results"].update(disabled=True)
            window['-IMAGE_RIGHT_TEXT-'].update("")
            window['-IMAGE_LEFT_TEXT-'].update("")
            window["-IMAGE0-"].update(visible=False)
            window["-IMAGE1-"].update(visible=False)
            
            
            
    # Close GUI if loop ends
    window.close()


if __name__ == "__main__":
    main()