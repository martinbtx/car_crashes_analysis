import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sklearn 


PATH_GLOBAL = ''

#Load useful tables from the dataset 
def load_dataset(batch_size) : 
    crash_location = pd.read_csv(''+PATH_GLOBAL+'ACCIDENT_LOCATION.csv',delimiter = ',',header=0 )
    crash_location = crash_location[["ACCIDENT_NO","NODE_ID"]]

    vehicle = pd.read_csv(''+PATH_GLOBAL+'VEHICLE.csv',delimiter = ',',header=0 )
    vehicle = vehicle[["ACCIDENT_NO", "TRAFFIC_CONTROL","VEHICLE_TYPE","Vehicle Type Desc"]]

    accident = pd.read_csv(''+PATH_GLOBAL+'ACCIDENT.csv',delimiter = ',',header=0 )
    accident['hour'] = pd.to_datetime(accident['ACCIDENTTIME']).dt.hour
    accident = accident[['ACCIDENT_NO', 'ACCIDENTDATE','hour','DAY_OF_WEEK',"LIGHT_CONDITION","Light Condition Desc",'NO_OF_VEHICLES','NO_PERSONS',"NO_PERSONS_INJ_2","NO_PERSONS_INJ_3","NO_PERSONS_KILLED","NO_PERSONS_NOT_INJ","ROAD_GEOMETRY","Road Geometry Desc","SEVERITY","SPEED_ZONE" ]]

    region = pd.read_csv(''+PATH_GLOBAL+'ACCIDENT_LOCATION.csv',delimiter = ',',header=0 )
    region = region[['ACCIDENT_NO','ROAD_TYPE']]

    atm_cond = pd.read_csv(''+PATH_GLOBAL+'ATMOSPHERIC_COND.csv',delimiter = ',',header=0 )
    atm_cond = atm_cond[["ACCIDENT_NO","ATMOSPH_COND","Atmosph Cond Desc"]]

    road_cond = pd.read_csv(''+PATH_GLOBAL+'ROAD_SURFACE_COND.csv',delimiter = ',',header=0 )
    road_cond = road_cond[["ACCIDENT_NO","SURFACE_COND","Surface Cond Desc"]]

    #Merging the tables and pre-selecting columns of interest 
    merge = pd.merge(accident,region,on='ACCIDENT_NO')
    merge = pd.merge(merge,vehicle,on='ACCIDENT_NO')
    merge = pd.merge(merge,atm_cond,on='ACCIDENT_NO')
    merge = pd.merge(merge,road_cond,on='ACCIDENT_NO')
    merge = pd.merge(merge,crash_location,on='ACCIDENT_NO')

    #Filtering out crashes older than 2018 to reduce the size of the dataset 
    merge['year'] = pd.to_datetime(merge['ACCIDENTDATE']).dt.year
    data_2018 = merge[merge['year'] > 2018]
    data_2018 = data_2018.dropna()
    #Final select of all the features of interest for the model --> the output is the nb of person injured + nb of person killed 
    final = data_2018[["NODE_ID","TRAFFIC_CONTROL","SPEED_ZONE","LIGHT_CONDITION","ATMOSPH_COND","SURFACE_COND","DAY_OF_WEEK","hour","NO_PERSONS_INJ_2","NO_PERSONS_INJ_3","NO_PERSONS_KILLED"]]
    final['accidents'] = final['NO_PERSONS_INJ_2']+final['NO_PERSONS_INJ_3']+final['NO_PERSONS_KILLED']
    final = final.drop(['NO_PERSONS_INJ_2', 'NO_PERSONS_INJ_3','NO_PERSONS_KILLED'], axis=1)

    #Discriminating inputs and targets to create the working dataset 
    col_nb = final.shape[1]
    final_inputs = sklearn.preprocessing.normalize(final.iloc[:,:col_nb-1],axis=0)
    final_labels = final['accidents'].to_numpy()

    dataset = CustomDataset(final_inputs,final_labels)

    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    data_train = DataLoader(train, batch_size = batch_size,shuffle=True)
    data_test = DataLoader(test,batch_size=batch_size,shuffle=False)

    return data_train, data_test

# train, test = load_dataset(12)
# a = next(iter(train))
# print(a)
