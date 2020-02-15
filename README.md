# Woody Cover estimation using Sentinel-1 time series in the Kruger National Park

**Student Assignment of GEO 419 - Modular Programming with Python for Application in Remote Sensing**

29.02.2020: Submission of Student Assignment <br>
Person in charge: Resha C.Y. Wibowo (resha.wibowo@uni-jena.de), <br> Lecturers for GEO 419 in charge: Martin Habermeyer, John Truckenbrodt, Prof. Dr. Christiane Schmullius

Task lists:
- [x] Derive woody cover information in the savanna ecosystem of the Kruger National Park from
Sentinel-1 and LIDAR data using machine learning
- [x] Accuracy assessment and comparison of two different machine learning approaches
- [x] Comparison of woody cover information from different time steps supporting the development of
a savanna ecosystem woody cover monitoring system
- [x] Interpretation of classification results

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
        dataset1 = os.path.join(default_dataset, dataset[1])
        dataset2 = os.path.join(default_dataset, dataset[2])

_Using module <font color="blue">callset()</font>, one reference (LiDAR data) and one dataset (Sentinel-1 Backscatter) can be called_

    ref1, meta_ref, dim_ref, pr_ref, band_ref, set1, meta_set, dim_set, pr_set, band_set = callset(wdataset,dataset1) # Call the dataset using <font color="blue">callset</font> module

    # Saving set1, since we have two Sentinel-1 datasets
    ds1 = {'set': set1,
           'meta': meta_set,
           'dim': dim_set,
           'prj': pr_set,
           'band': band_set}

    del ref1, meta_ref, dim_ref, pr_ref, band_ref, set1, meta_set, dim_set, pr_set, band_set # Clear Memory

    # Second dataset
    ref1, meta_ref, dim_ref, pr_ref, band_ref, set1, meta_set, dim_set, pr_set, band_set = callset(wdataset,dataset2) # Call the dataset using <font color="blue">callset</font> module

    # Saving set2
    ds2 = {'set': set1,
           'meta': meta_set,
           'dim': dim_set,
           'prj': pr_set,
           'band': band_set}

    df_ref = ref1.ReadAsArray() # Register Reference dataframe
    df_set1 = ds1["set"].ReadAsArray() # Register 1. Sentinel dataframe
    df_set2 = ds2["set"].ReadAsArray() # Register 2. Sentinel dataframe


_Printing Spatial Properties of Dataframes_

    print("LiDAR Data:", str(dataset[0]))
    print("Projection: ", str(pr_ref[8:28]))
    print("Properties:")
    print("Dimension: ", dim_ref[0], "x", dim_ref[1])
    print("Size: ", np.size(df_ref))
    print("Layer(s): ", band_ref)
    print("----------")
    print("SAR Data:", str(dataset[1]), "and", str(dataset[2]))
    print("Projection: ", str(ds1["prj"][8:28]), "and", str(ds2["prj"][8:28]))
    print("Properties:")
    print("Dimension: ", ds1["dim"][0], "x", ds1["dim"][1], "and", ds2["dim"][0], "x", ds2["dim"][1])
    print("Layer(s): ", ds1["band"], "and", ds2["band"])
    print("Size: ", np.size(df_set1), "and", np.size(df_set2))


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
    tsize = 0.7 # set test size for machine learning split-train-test
    estimator = 100 # number of estimator for RandomForest in range 10 to 500

    file_name = "Woody" # reference filename
    heading_name = "LiDAR Woody Cover" # heading on figure

    file_names = "S1_TS_17_18" # dataset filename
    heading_names = "Sentinel-1 Timeseries 2017/2018" # heading on figure

    file_names2 = "S1_TS_18_19" # dataset filename
    heading_names2 = "Sentinel-1 Timeseries 2018/2019" # heading on figure

    img_slc1 = ds1["set"].GetRasterBand(selection).ReadAsArray() # store for prediction map
    img_slc2 = ds2["set"].GetRasterBand(selection).ReadAsArray() # store for prediction map

    kwargs = {'datamap': '',
              'datainput': '',
              'filename': '',
              'headingname': '',
              'selection': selection, #by default
              'output': '',
              'method':'',
              'nestimator': estimator} # Default kwargs for saving figure later

_Using module <font color="blue">istore()</font>, the dataset's Figure can be generated, by certain set (kwargs), then stored into exact desired folder_
          
**5. Set output folders, for figure storage**

    OM = os.path.join(default_output,'Overview_Map/') # set output dir for overview map
    SUB = os.path.join(default_output,'Subset/') # set output dir for subset map
    PRD = os.path.join(default_output,'Prediction/') # set output dir for prediction map
    DIF = os.path.join(default_output,'Dif_Map/') #set output dir for comparison map

    try: # Check the availability of each folder
        os.mkdir(OM)
        os.mkdir(SUB)
        os.mkdir(PRD)
        os.mkdir(DIF)
    except OSError:
        os.rmdir(OM)
        os.rmdir(SUB)
        os.rmdir(PRD)
        os.rmdir(DIF)
        os.mkdir(OM)
        os.mkdir(SUB)
        os.mkdir(PRD)
        os.mkdir(DIF)
    else:
        os.makedirs(OM,exist_ok=True)
        os.makedirs(SUB,exist_ok=True)
        os.makedirs(PRD,exist_ok=True)
        os.makedirs(DIF,exist_ok=True)

### Saving Overview Map, creating Dataset-subset and saving Subset Map

**6.1. Save overview map**

    kwargs['datamap'], kwargs['output'] = 'om', OM # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'woody', file_name, heading_name
    istore(df_ref,**kwargs) # Save Reference Map

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    istore(df_set1,**kwargs) # Save Dataset Map

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names2, heading_names2
    istore(df_set2,**kwargs) # Save Dataset Map
    
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

    img_ref = class_ref.copy()

**6.2. Create dataset subset, save subset map**

_Using module <font color="blue">subsets()</font>, the reference (LiDAR data) will be used as extend and the dataset (Sentinel-1 Backscatter) will be clipped by extend_

    xoff, yoff, xcount, ycount = subsets(ref1, ds1["set"]) # Clipped the dataset using <font color="blue">subsets</font> module
    img_set1 = ds1["set"].GetRasterBand(selection).ReadAsArray(xoff, yoff, xcount, ycount) # Clipped data

    del xoff, yoff, xcount, ycount # clear memory

    xoff, yoff, xcount, ycount = subsets(ref1, ds2["set"]) # Clipped the dataset using <font color="blue">subsets</font> module
    img_set2 = ds2["set"].GetRasterBand(selection).ReadAsArray(xoff, yoff, xcount, ycount) # Clipped data

    kwargs['datamap'], kwargs['output'] = 'sub', SUB # set category and outputmap

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names, heading_names
    istore(img_set1,**kwargs) # Saving clipped Dataset

    kwargs['datainput'],kwargs['filename'],kwargs['headingname'] = 'sentinel', file_names2, heading_names2
    istore(img_set2,**kwargs) # Saving clipped Dataset

### Test and Sampling

**7. Do split train and test**

    ref_train, ref_test = train_test_split(img_ref, test_size=tsize,random_state=rd_state, shuffle=False)

    set1_train, set1_test, set2_train, set2_test = train_test_split(img_set1, img_set2,
                                                                    test_size=tsize,
                                                                    random_state=rd_state, shuffle=False)

    mask = ~np.isnan(set1_train) & ~np.isnan(set2_train) & ~np.isnan(ref_train) # creating ~NAN Mask
    mask2 = ~np.isnan(set1_test) & ~np.isnan(set2_test) & ~np.isnan(ref_test) # creating ~NAN Mask

    set1_train_mask = set1_train[mask].reshape(-1, 1)
    set2_train_mask = set2_train[mask].reshape(-1, 1) # Masking split_train datasets
    ref_train_mask = ref_train[mask] # Masking split_train reference

    set1_test = set1_test[mask2].reshape(-1,1)
    set2_test = set2_test[mask2].reshape(-1,1) # Masking for Prediction Machine Learning
    ref_test_mask = ref_test[mask2] # Masking split_train reference


### Instantiation of the Model

**8. Set Machine Learning module**

    Kernel_SVC_model1 = svm.SVC(decision_function_shape='ovo', random_state = rd_state)
    Kernel_SVC_model2 = svm.SVC(decision_function_shape='ovo', random_state = rd_state)
    RandomForest_model1 = RandomForestClassifier(n_estimators=estimator, random_state = rd_state)
    RandomForest_model2 = RandomForestClassifier(n_estimators=estimator, random_state = rd_state)

### Fitting of the Classifiers

**9. Fit split-train set data and reference data**

    Kernel_SVC_model1.fit(set1_train_mask, ref_train_mask)
    Kernel_SVC_model2.fit(set2_train_mask, ref_train_mask)
    RandomForest_model1.fit(set1_train_mask, ref_train_mask)
    RandomForest_model2.fit(set2_train_mask, ref_train_mask)

### Prediction and Accuracy Assessment

**10.1. Split-test prediction**

    Kernel_SVC_prediction1 = Kernel_SVC_model1.predict(set1_test)
    Kernel_SVC_prediction2 = Kernel_SVC_model2.predict(set2_test)
    RandomForest_prediction1 = RandomForest_model1.predict(set1_test)
    RandomForest_prediction2 = RandomForest_model2.predict(set2_test)

**10.2. Accuracy of Machine Learning**

_Support Vector Machine_

    Accuracy1 = {"Accuracy_SVC": metrics.accuracy_score(ref_test_mask,Kernel_SVC_prediction1),
                  "Matrix_SVC": metrics.confusion_matrix(ref_test_mask, Kernel_SVC_prediction1),
                  "Report_SVC": metrics.classification_report(ref_test_mask, Kernel_SVC_prediction1)}

    Accuracy2 = {"Accuracy_SVC": metrics.accuracy_score(ref_test_mask,Kernel_SVC_prediction2),
              "Matrix_SVC": metrics.confusion_matrix(ref_test_mask, Kernel_SVC_prediction2),
              "Report_SVC": metrics.classification_report(ref_test_mask, Kernel_SVC_prediction2)}

_Random Forest_

    Accuracy1["Accuracy_RF"], Accuracy1["Matrix_RF"], Accuracy1["Report_RF"]  =  metrics.accuracy_score(ref_test_mask, RandomForest_prediction1),metrics.confusion_matrix(ref_test_mask, RandomForest_prediction1), metrics.classification_report(ref_test_mask, RandomForest_prediction1)
    Accuracy2["Accuracy_RF"], Accuracy2["Matrix_RF"], Accuracy2["Report_RF"]  =  metrics.accuracy_score(ref_test_mask, RandomForest_prediction2),metrics.confusion_matrix(ref_test_mask, RandomForest_prediction2), metrics.classification_report(ref_test_mask, RandomForest_prediction2) 

Printing accuracy, confusion matrix, and classification report

    print("Model data", file_names)
    print("Support Vector Machine")
    print("Accuracy:", Accuracy1["Accuracy_SVC"])
    print(Accuracy1["Matrix_SVC"])
    print(Accuracy1["Report_SVC"])
    print("")
    print("Random Forest")
    print("Accuracy:", Accuracy1["Accuracy_RF"])
    print(Accuracy1["Matrix_RF"])
    print(Accuracy1["Report_RF"])
    print("""

    """)
    print("Model data", file_names2)
    print("Support Vector Machine")
    print("Accuracy:", Accuracy2["Accuracy_SVC"])
    print(Accuracy2["Matrix_SVC"])
    print(Accuracy2["Report_SVC"])
    print("")
    print("Random Forest")
    print("Accuracy:", Accuracy2["Accuracy_RF"])
    print(Accuracy2["Matrix_RF"])
    print(Accuracy2["Report_RF"])

**10.3. Predicting dataset imageries**

    Prediction1_SVC = Kernel_SVC_model1.predict(img_slc1.flatten().reshape(-1,1))
    Prediction2_SVC = Kernel_SVC_model2.predict(img_slc2.flatten().reshape(-1,1))
    Prediction1_RF = RandomForest_model1.predict(img_slc1.flatten().reshape(-1,1))
    Prediction2_RF = RandomForest_model2.predict(img_slc2.flatten().reshape(-1,1))


**10.4. Saving image prediction**

    kwargs['datamap'], kwargs['output'] = 'prd', PRD # set category and outputmap
    kwargs['method'] = 'SVC'
    kwargs['filename'], kwargs['accuracy'] = file_names, round(Accuracy1["Accuracy_SVC"],2)
    sets = Prediction1_SVC.reshape(img_slc1.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving SVC prediction
    sets = 0 # clear memory

    kwargs['method'] = 'RF'
    kwargs['accuracy'] = round(Accuracy1["Accuracy_RF"],2)
    sets = Prediction1_RF.reshape(img_slc1.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving RandomForest prediction
    sets = 0 # clear memory

    kwargs['method'] = 'SVC'
    kwargs['filename'], kwargs['accuracy'] = file_names2, round(Accuracy2["Accuracy_SVC"],2)
    sets = Prediction2_SVC.reshape(img_slc2.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving SVC prediction
    sets = 0 # clear memory

    kwargs['method'] = 'RF'
    kwargs['accuracy'] = round(Accuracy2["Accuracy_RF"],2)
    sets = Prediction2_RF.reshape(img_slc2.shape) # since prediction's shape is 1D, we need to reshape our array to 2D
    set0 = sets < 0
    sets[set0] = 0
    istore(sets,**kwargs) # Saving RandomForest prediction
    sets = 0 # clear memory
    
**10.4. Machine Learning Assessment and Time-series Comparison**

Calculating Accuracy Different from Machine Learning

    d1 = abs(Accuracy1["Accuracy_RF"] - Accuracy1["Accuracy_SVC"])
    d2 = abs(Accuracy2["Accuracy_RF"] - Accuracy2["Accuracy_SVC"])

Different Map

    d1_map = Prediction1_RF - Prediction1_SVC
    d2_map = Prediction2_RF - Prediction2_SVC
    dSVC_map = Prediction1_SVC - Prediction2_SVC 
    dRF_map = Prediction1_RF  - Prediction2_RF
    
Saving Different Map

    kwargs['datamap'], kwargs['output'] = 'dif', DIF # set category and outputmap
    kwargs['datainput'] = 'method'
    
    kwargs['filename'],kwargs['headingname'], kwargs['accuracy'] = file_names, heading_names, d1
    istore(d1_map.reshape(img_slc1.shape), **kwargs) # Saving ML Assesment 1
    
    kwargs['filename'],kwargs['headingname'], kwargs['accuracy'] = file_names2, heading_names2, d2
    istore(d2_map.reshape(img_slc1.shape), **kwargs) # Saving ML Assesment 2
    
    kwargs['datainput'] = 'time'
    kwargs['method'] = 'SVC'
    kwargs['filename'],kwargs['headingname'] = 'time_series', '2017/2018 to 2018/2019'
    istore(dSVC_map.reshape(img_slc1.shape),**kwargs) # Saving Different SVC Prediction

    kwargs['method'] = 'RF'
    kwargs['filename'],kwargs['headingname'] = 'time_series', '2017/2018 to 2018/2019'
    istore(dRF_map.reshape(img_slc1.shape),**kwargs) # Saving Different RF Prediction

    del d1_map, d2_map, dSVC_map, dRF_map # clear memory
