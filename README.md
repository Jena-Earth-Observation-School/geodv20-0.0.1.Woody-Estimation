# Woody Cover estimation using Sentinel-1 time series in the Kruger National Park

**Student Assignment of GEO 419 - Modular Programming with Python for Application in Remote Sensing**

29.02.2020: Submission of Student Assignment
Person in charge: Resha C.Y. Wibowo (resha.wibowo@uni-jena.de), <br> Lecturers for GEO 419 in charge: Martin Habermeyer, John Truckenbrodt, Prof. Dr. Christiane Schmullius

Task lists:
- [x] derive woody cover information in the savanna ecosystem of the Kruger National Park from
Sentinel-1 and LIDAR data using machine learning
- [x] accuracy assessment and comparison of two different machine learning approaches
- [ ] comparison of woody cover information from different time steps supporting the development of
a savanna ecosystem woody cover monitoring system
- [ ] Interpretation of classification results

Our basic concept on subset processes through imageries is insipired by https://geohackweek.github.io/raster/04-workingwithrasters/, however we would like to implement Gdal to generate our analysis (One of our main code: https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html). <br>
References used in this system are tagged in the code to give credits to the owner of the code.

**Our approaches in this system are by using SVM (Support Vector Machine), and RF (Random Forest)**


### Load all required modules into the system
It's is important to know which modules are necessary for our system and to call all of the essential modules on the beginning of our system to minimise minor error at the middle of the framework.


**1. Calling all required moduls for our processes. We divide our modules into: General, Raster Calling, and Machine Learning**

**General Modules**

    %matplotlib qt
    import os
    import numpy as np
    from scipy import io
    import pandas as pd
    import seaborn as sns
    import pyproj


**Raster Calling Modules (gdal)** https://pypi.org/project/GDAL/

    from osgeo import gdal, ogr, osr, gdal_array, gdalconst, sys
    gdal.UseExceptions() # Tell GDAL to throw Python exceptions, and register all drivers (https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html)
    gdal.AllRegister()

**Import personal modules**

    from mlp import callset,cref,istore,subsets


**Import Machine learning (sklearn) modules**

_used as accuracy assessment and train-test split respectively_

    from sklearn import metrics
    from sklearn.model_selection import train_test_split

_Support Vector Machine (_ https://scikit-learn.org/stable/modules/svm.html _)_
    
    from sklearn import svm

_Random Forest Classifier (_ https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier _)_

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification


### Settings of system directory and calling datasets

**2. Set all directory to be used**
Since working with directory based modules such as Gdal, one should consider to organise the directory before doing works. Either being set or not, this steps are necessary to be done.

    os.getcwd() # checking directory of working

##### Set the default directory
    default_directory = "/home/geodv20/Desktop/Friedrich-Schiller Uni Jena/Material/Semester1/GEO419/Abschlussaufgabe"
    default_data = "/home/geodv20/Desktop/Friedrich-Schiller Uni Jena/Material/Semester1/GEO419/Abschlussaufgabe/data"
    default_dataset = "/home/geodv20/Desktop/Friedrich-Schiller Uni Jena/Material/Semester1/GEO419/Abschlussaufgabe/data/xx_419"
    default_output = "/home/geodv20/Desktop/Friedrich-Schiller Uni Jena/Material/Semester1/GEO419/Abschlussaufgabe/output"

    print("""
    Tracing files inside data directory

    """) 
    for root, dirs, files in os.walk(default_data):
        for file in files:
            print(os.path.join(root, file)) # show any files inside the current directory  sc: https://www.tutorialspoint.com/python/os_walk.htm

**3. Calling datasets**
Our datasets used in this system are:
- 02_KNP_LIDAR_DSM_50m_woody_cover_perc;
- S1_A_D_VH_VV_stack_04_2017_04_2018_KNP; and
- S1_A_D_VH_VV_stack_04_2018_04_2019_KNP.

    dataset = ("02_KNP_LIDAR_DSM_50m_woody_cover_perc", "S1_A_D_VH_VV_stack_04_2017_04_2018_KNP", "S1_A_D_VH_VV_stack_04_2018_04_2019_KNP")
    wdataset = os.path.join(default_dataset, dataset[0])
    s2017 = os.path.join(default_dataset, dataset[1])
    s2018 = os.path.join(default_dataset, dataset[2])

_Using module <font color="blue">callset()</font>, one reference (LiDAR data) and one dataset (Sentinel-1 Backscatter) can be called_

    ref1, meta_ref, dim_ref, pr_ref, band_ref, set1, meta_set, dim_set, pr_set, band_set = callset(wdataset,s2017) # Call the dataset using <font color="blue">callset</font> module

    df_ref = ref1.ReadAsArray() # Register Reference dataframe
    df_set = set1.ReadAsArray() # Register Sets dataframe

_Printing Spatial Properties of Dataframes_

    print("LiDAR Data:", str(dataset[0]))
    print("Projection: ", str(pr_ref[8:28]))
    print("Properties:")
    print("Dimension: ", dim_ref[0], "x", dim_ref[1])
    print("Size: ", np.size(df_ref))
    print("Layer(s): ", band_ref)
    print("----------")
    print("SAR Data:", str(dataset[1]))
    print("Projection: ", str(pr_set[8:28]))
    print("Properties:")
    print("Dimension: ", dim_set[0], "x", dim_set[1])
    print("Layer(s): ", band_set)
    print("Size: ", np.size(df_set))

_Using module <font color="blue">cref()</font>, classes in reference (LiDAR data) can be identified_

    n_ref, c_ref, dict_ref = cref(df_ref) # Seek classes over reference dataframe

    print('We have {n} Woody References'.format(n=n_ref))
    print("""The training data include {n} classes (%coverages):
        {classes}""".format(n=c_ref.size, 
                            classes=c_ref))
    print(dict_ref) # Print classes

**4. Specify data for following process**

    selection = 24 # selected time-series layer
    rd_state = 204 # set random_state for machine learning, and split-train-test module

    file_name = "Woody" # reference filename
    heading_name = "LiDAR Woody Cover" # heading on figure

    file_names = "S1_TS_17_18" # dataset filename
    heading_names = "Sentinel-1 Timeseries 2017/2018" # heading on figure

    img_slc = set1.GetRasterBand(selection).ReadAsArray() # store for prediction map

    kwargs = {'datamap': '',
              'datainput': '',
              'filename': '',
              'headingname': '',
              'selection': selection, #by default
              'output': '',
              'method':''} # Default kwargs for saving figure later

_Using module <font color="blue">istore()</font>, the dataset's Figure can be generated, by certain set (kwargs), then stored into exact desired folder_
          
**5. Set output folders, for figure storage**

    OM = os.path.join(default_output,'Overview_Map/') #set output dir for overview map
    SUB = os.path.join(default_output,'Subset/') #set output dir for subset map
    TEST = os.path.join(default_output,'Test/') #set output dir for test map
    TRAIN = os.path.join(default_output,'Train/') #set output dir for train map
    PRD = os.path.join(default_output,'Prediction/') #set output dir for prediction map

    try: # Check the availability of each folder
        os.mkdir(OM)
        os.mkdir(SUB)
        os.mkdir(TEST)
        os.mkdir(TRAIN)
        os.mkdir(PRD)
    except OSError:
        os.rmdir(OM)
        os.rmdir(SUB)
        os.rmdir(TEST)
        os.rmdir(TRAIN)
        os.rmdir(PRD)
        os.mkdir(OM)
        os.mkdir(SUB)
        os.mkdir(TEST)
        os.mkdir(TRAIN)
        os.mkdir(PRD)
    else:
        os.makedirs(OM,exist_ok=True)
        os.makedirs(SUB,exist_ok=True)
        os.makedirs(TEST,exist_ok=True)
        os.makedirs(TRAIN,exist_ok=True)
        os.makedirs(PRD,exist_ok=True)

### Saving Overview Map, creating Dataset-subset and saving Subset Map

**6.1. Save overview map**

    kwargs['datamap'], kwargs['output'] = 'om', OM # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'woody', file_name, heading_name
    istore(df_ref,**kwargs) # Save Reference Map

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    istore(df_set,**kwargs) # Save Dataset Map

_Classify woody cover into 10 classes by range 10% (0-10%, 10-20%, and so on)_

    class_ref = df_ref.copy() # Make sure main dataframe remained

    class_0 = class_ref < 0
    class_1 = np.bitwise_and(class_ref >= 0, class_ref <= 10)
    class_2 = np.bitwise_and(class_ref > 10, class_ref <= 20)
    class_3 = np.bitwise_and(class_ref > 20, class_ref <= 30)
    class_4 = np.bitwise_and(class_ref > 30, class_ref <= 40)
    class_5 = np.bitwise_and(class_ref > 40, class_ref <= 50)
    class_6 = np.bitwise_and(class_ref > 50, class_ref <= 60)
    class_7 = np.bitwise_and(class_ref > 60, class_ref <= 70)
    class_8 = np.bitwise_and(class_ref > 70, class_ref <= 80)
    class_9 = np.bitwise_and(class_ref > 80, class_ref <= 90)
    class_10 = np.bitwise_and(class_ref > 90, class_ref <= 100)

    class_ref[class_0] = "NaN"
    class_ref[class_1] = 1
    class_ref[class_2] = 2
    class_ref[class_3] = 3
    class_ref[class_4] = 4
    class_ref[class_5] = 5
    class_ref[class_6] = 6
    class_ref[class_7] = 7
    class_ref[class_8] = 8
    class_ref[class_9] = 9
    class_ref[class_10] = 10

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'woody', 'Classified_Woody', 'Classified Woody'
    istore(class_ref,**kwargs) # Saving classified woody reference

    class_ref[np.isnan(class_ref)] = -99 # Since following process will get issues with NAN values
    img_ref = class_ref.copy()

**6.2. Create dataset subset, save subset map**

_Using module <font color="blue">subsets()</font>, the reference (LiDAR data) will be used as extend and the dataset (Sentinel-1 Backscatter) will be clipped by extend_

    xoff, yoff, xcount, ycount = subsets(ref1, set1) # Clipped the dataset using <font color="blue">subsets</font> module
    img_set = set1.GetRasterBand(selection).ReadAsArray(xoff, yoff, xcount, ycount) # Clipped data

    kwargs['datamap'], kwargs['output'] = 'sub', SUB # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    istore(img_set,**kwargs) # Saving clipped Dataset

### Test and Sampling

**7. Do split train and test**

    ref_train, ref_test, set_train, set_test = train_test_split(img_ref, img_set,
                                                                test_size=0.7,
                                                                random_state=rd_state, shuffle=False)

    mask = ~np.isnan(set_train) & ~np.isnan(ref_train) # creating ~NAN Mask

    set_train_mask = set_train[mask].reshape(-1, 1) # Masking split_train datasets
    set_test = set_test.flatten().reshape(-1, 1) # Flatten Array for Prediction Machine Learning
    ref_train_mask = ref_train[mask] # Masking split_train reference

**7.1. Saving Test Map**

    kwargs['datamap'], kwargs['output'] = 'test', TEST  # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'woody', file_name, heading_name
    istore(ref_test,**kwargs) # Save Reference Test

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    sets = set_test.reshape(ref_test.shape) #since masked set change the shape, we reshape our array
    istore(sets,**kwargs) # Save Dataset Test
    sets = 0 # clear memory

**7.2. Saving Train Map**

    kwargs['datamap'], kwargs['output'] = 'train', TRAIN # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'woody', file_name, heading_name
    istore(ref_train,**kwargs) # Save Reference Test

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    sets = set_train_mask.reshape(set_train.shape) #since masked set change the shape, we reshape our array
    istore(sets,**kwargs) # Save Dataset Test
    sets = 0 # clear memory

### Instantiation of the Model

**8. Set Machine Learning module**

    Kernel_SVC_model = svm.SVC(random_state = rd_state)
    RandomForest_model = RandomForestClassifier(n_estimators=24, random_state = rd_state)

### Fitting of the Classifiers

**9. Fit split-train set data and reference data**

    Kernel_SVC_model.fit(set_train_mask, ref_train_mask)
    RandomForest_model.fit(set_train_mask, ref_train_mask)

### Prediction and Accuracy Assessment

**10.1. Split-test prediction**

    Kernel_SVC_prediction = Kernel_SVC_model.predict(set_test)
    RandomForest_prediction = RandomForest_model.predict(set_test)

**10.2. Accuracy of Machine Learning**

_Support Vector Machine_

    Kernel_SVC_accuracy = metrics.accuracy_score(ref_test.flatten(),Kernel_SVC_prediction)
    Kernel_SVC_confusion_matrix = metrics.confusion_matrix(ref_test.flatten(), Kernel_SVC_prediction)
    Kernel_SVC_classification_report = metrics.classification_report(ref_test.flatten(), Kernel_SVC_prediction)
    print("Accuracy:", Kernel_SVC_accuracy)
    print(Kernel_SVC_confusion_matrix)
    print(Kernel_SVC_classification_report)

_Random Forest_

    RandomForest_accuracy = metrics.accuracy_score(ref_test.flatten(), RandomForest_prediction)
    RandomForest_confusion_matrix = metrics.confusion_matrix(ref_test.flatten(), RandomForest_prediction)
    RandomForest_classification_report = metrics.classification_report(ref_test.flatten(), RandomForest_prediction)
    print("Accuracy:", RandomForest_accuracy)
    print(RandomForest_confusion_matrix)
    print(RandomForest_classification_report)

**10.3. Saving Prediction Map**

    kwargs['datamap'], kwargs['output'] = 'prd', PRD # set category and outputmap
    kwargs['datainput'] = '' # clear memory, since we don't need datainput for following lines

    kwargs['method'] = 'SVC'
    kwargs['filename'],kwargs['headingname'], kwargs['accuracy'] = 'Set_test', heading_names, round(Kernel_SVC_accuracy,2)
    sets = Kernel_SVC_prediction.reshape(ref_test.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    istore(sets,**kwargs) # Saving SVC prediction
    sets = 0 # clear memory

    kwargs['method'] = 'RF'
    kwargs['accuracy'] = round(RandomForest_accuracy,2)
    sets = RandomForest_prediction.reshape(ref_test.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    istore(sets,**kwargs) # Saving RandomForest prediction
    sets = 0 # clear memory

**10.4. Predicting rest of image**

    Prediction_SVC = Kernel_SVC_model.predict(img_slc.flatten().reshape(-1,1))
    Prediction_RF = RandomForest_model.predict(img_slc.flatten().reshape(-1,1))

**10.5. Saving whole image prediction**

    kwargs['method'] = 'SVC'
    kwargs['filename'], kwargs['accuracy'] = file_names, round(Kernel_SVC_accuracy,2)
    sets = Prediction_SVC.reshape(img_slc.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving SVC prediction
    sets = 0 # clear memory

    kwargs['method'] = 'RF'
    kwargs['accuracy'] = round(RandomForest_accuracy,2)
    sets = Prediction_RF.reshape(img_slc.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving RandomForest prediction
    sets = 0 # clear memory

