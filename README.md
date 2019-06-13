# prco304-final-year-project-erharrison
prco304-final-year-project-erharrison created by GitHub Classroom

HOW TO EXECUTE THIS APPLICATION

1. Once the project has been downloaded, it needs to be unzipped.

2. Install Python 3.6.7

3. Set the project interpreter to be the python.exe that is associated with the installed Python

4. Before the application can be run successfully the following libraries and their specific versions need to be installed 
(I recommend installing pip first, then installing everything else using pip):
- Jinja2	2.10.1
- Keras	2.2.4
- Keras-Applications	1.0.7
- Keras-Processing	1.0.9
- MarkupSafe	1.1.1
- PyYAML	5.1
- Werkzeug	0.15.2
- XlsxWriter	1.1.7
- absl-py	0.7.1
- astor	0.7.1
- attrs	19.1.0
- branca	0.3.1
- certifi	2019.3.9
- chardet	3.0.4
- cvxpy	1.0.21
- cycler	0.10.0
- dill	0.2.9
- ecos	2.0.7.post1
- fastcache	
- future	0.17.1
- folium	0.8.3
- gast	0.2.2
- graphviz	0.10.1 (Graphviz is used for plotting the model and also needs to be installed via a web browser for it to work.)
- grpcio	1.20.0
- h5py	2.9.0
- idna	2.8
- incremental		17.5.0
- joblib	0.13.2
- kiwisolver	1.0.1
- layers	0.1.5
- matplotlib	3.0.3
- mock	2.0.0
- multiprocess	0.70.7
- np-utils	0.5.10.0
- numpy	1.16.2
- pandas	0.24.2
- pbr	5.1.3
- pip	10.0.1
- protobuf	3.7.1
- pydot	1.4.1
- pyparsing	2.4.0
- python-dateutil	2.8.0
- pytz	2019.1
- requests	2.21.0
- rnn	0.0.0
- scikit-keras	0.1.10
- scikit-learn	0.20.3
- scipy	1.2.1
- scs	2.1.0
- setuptools	39.0.1
- six	1.12.0
- tensorboard	1.13.1
- tensorflow	1.13.1
- tensorflow -estimator	1.13.1
- termcolor	1.1.0
- urllib3	1.24.2
- wheel	0.33.1
- xlrd	1.2.0
- SALib	1.3.4

4. Exchange the file paths in the code for ones that are appropriate for your device.
   (The dataset is contained within the file 'ImputedData.xlsx' held within this repository.
   The coordinates data is within the file 'Coordinates.xlsx', also held within this repository.)
   
5. Run the file NN.py

6. At the end of the applications's execution it will prompt for user input in the Python console.

7. After this has been entered the bubble map is generated, this has to be manually opened

8. To view TensorBoard, after the map has been generated:
  
  8.1 this command has to be run: tensorboard --logdir=path/to/log-directory
    (The directory path has be exhanged for the one the code points towards.)
  
  8.2 Access localhost:6006 in your web browser, and voila!


