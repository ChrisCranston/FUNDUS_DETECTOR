# FUNDUS_DETECTOR

##--RUNNING THE APPLICATION-- 

After extracting files the application can be run in one of two ways:
1) Command line

`cd Application`
`pip install -r requirements.txt`
`python -B ./main.py`

### NOTE:
first run of the app can take a few second to load.

2) Packaged application

`unzip the included FUNDUS_APPLICATION.zip file`
`move into FUNDUS_APPLICATION folder`
`Double click FUNDUS_APPLICATION.exe` or `drag FUNDUS_APPLICATION.exe into cmd and hit enter`

### NOTE: 
the unzipped folder of the application is approximately 3.5Gb due to the use of tensorflow and torch.
      Any issues with the zip file a second copy is available here:
		https://drive.google.com/file/d/16KjYg65rOsaqpiaolQvllrstkcD4gtLx/view?usp=sharing 

the first run of the application using the packaged files takes approximately 30 seconds. 




## --TEST DATA--

Test data has been included in the /test-data file, each folder contains a .text file used to load the images. Upon inferencing the inferenced results are stored in the same folder under a /results subdirectory. 
The table below shows the expected results for each patient if the 0.20 default thresholding is used:

-----------------------------
|Patient # | Expected result |
|----------|-----------------|
| 1        | R0              |
| 2        | R2              |
| 3        | R2              |
| 4        | R1              |
| 5        | R0              |
| 6        | R0              |
| 7        | R0              |
| 8        | R1              |
-----------------------------

### NOTE:
The entirety of the image dataset used was not included in an effort to reduce size of zip, i attempted to capture a range of examples in the tests but the dataset is available at:
	https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
and additional tests can be created by copying a test patient file from /test-data/ replacing images, and updating the .txt file to address the new images.

	


## -- ADDITIONAL FILES -- 

The files included in `/data-pre-processing-matlab-python` are not required in any way for the application. 
These files were created to handle:
	- conversion of MATLAB image labeller struct to a usable python format through parquet
	- Prepping label format and folder layout for the training of the model 


## -- MODEL TRAINING AND FILES NOT CREATED BY ME --
To train the model the yolov5 official repository was used, the repo and instructions can be found here:
	 https://github.com/ultralytics/yolov5

within the final application some files were required when performing detections using YOLOv5, each file underwent some editing to make them application appropriate but for the most part they are directly from the above repository.
	- Contents of /utils folder 
	- Contents of /models folder

The primary files created by me are (comments added to top of these files where possible):
	- main.py
	- /detector/detect.py
	- /classifier/classify.py
	- contents of /data (created during model training)
	- contents of /input (created during model training)




## -- CORRUPT / ISSUES WITH ZIPPED FILES -- 

if for any reason there is an issue with the zipped files from submission a back-up of these files is available publicly at:
	https://github.com/ChrisCranston/FUNDUS_DETECTOR
