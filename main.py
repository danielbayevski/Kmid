

import argparse
import os.path
import time

import numpy
from sklearn_extra.cluster import KMedoids
import pandas as pd
import sys
import psutil
import math
DATABUFFER = 16*math.sqrt(psutil.virtual_memory().free) # 8 MB
# import datetime



def load_data(path,startrows=None,endrows=None,cols=None):
    # load_data(path, startRow, endRow, cols) -> data
    # path – path to file startRow – first row to load
    # endRow – last row to load, or the last of file, the minimum
    # cols – columns to load, there may be columns we would not want to load as they are useless
    # data – return the loaded partial file
    row_num = endrows-startrows if startrows else endrows
    try:
        if path[-5:] == ".xlsx" or path[-4:] == ".xls":
            header =pd.read_excel(path,usecols=cols,nrows=1)
            temp = pd.read_excel(path,usecols=cols,nrows=row_num,skiprows=startrows)
            temp.columns = header.columns
        elif path[-4:] == ".csv":
            header = pd.read_csv(path, usecols=cols, nrows=1)
            temp=pd.read_csv(path,usecols=cols,nrows=row_num,skiprows=startrows)
            temp.columns = header.columns
    except:
        temp = pd.DataFrame()

    return temp

def save_file(path, pred, startrows=None, endrows=None):
    # save_data(path, startRow, endRow)
    # path – path to the file to load to,
    #               or create a blank file, titled “path_new.csv/xls”
    # startRow – first row to save
    # endRow – last row to load, or the last of file, the minimum
    # TODO:find better way to partially save file,
    #  dont want to take too much memory
    temp = load_data(path, startrows, endrows)
    temp["grouping"] = pred.tolist()
    if path[-5:] == ".xlsx":
        new_path = path[:-5] + "_new" + path[-5:]
        temp.to_excel(new_path,mode='a')# temp.to_excel(new_path,index=[i for i in range(startrows,endrows)])
    elif path[-4:] == ".xls":
        new_path = path[:-4] + "_new" + path[-4:]
        temp.to_excel(new_path,mode='a')
    elif path[-4:] == ".csv":
        new_path = path[:-4] + "_new" + path[-4:]
        temp.to_csv(new_path,mode='a')


def decide_row_num_halt(path,cols=None):
    #decide how many rows can you work with at atime, depends on the DATABUFFER
    if path[-5:] == ".xlsx" or path[-4:] == ".xls":
        temp = pd.read_excel(path,usecols=cols,nrows=1)
    elif path[-4:] == ".csv":
        temp=pd.read_csv(path,usecols=cols,nrows=1)
    return round(DATABUFFER/ sys.getsizeof(temp))


def get_time_replace_reduct_cols(path,rowNum):
    #get_time_replace_reduct_cols(data) -> timeCols,replaceDict,colNums
    # data – data we work with currently
    # timeCols – all columns with timestamps
    # replaceDict  - list of columns and dictionaries to replace their values with
    # colNums – list of columns to load, as some of them are useless, example:  personal numbers, passwords and OTP

    data = load_data(path, endrows=rowNum)
    timeCols = []
    replaceDict = {}
    colNums = []
    numberTypes = numpy.array(["int8","int16","int32","int64","uint8","uint16","uint64","uint32","float32","float64"])
    colLen = len(data.axes[0])
    rowsForSingleColumn = decide_row_num_halt(path,cols=[0])
    for i,col in enumerate(data.columns):
        if data[col].dtype.name[:8]=="datetime":
            timeCols.append(col)
            colNums.append(i)

        #check for OTP
        elif (data[col].values.dtype== numberTypes).any() :
            colNums.append(i)
        else:
            colNums.append(i)
            p = 0
            data1 = load_data(path, endrows=rowsForSingleColumn, cols=[i])
            while not data1.empty:
                for col in data1.columns:
                    T = 0
                    F = 0
                    things = []
                    numbers = []
                    for thing in data1[col].unique():
                        if isinstance(thing,str):
                            try:
                                float(thing.replace(',', ''))
                                T+=1
                                numbers.append(thing)
                            except:
                                things.append(thing)
                                F += 1
                            continue
                        if (thing.dtype == numberTypes).any():
                            T += 1
                        else:
                            things.append(i)
                            F += 1
                    if T / (T + F) > 0.8:
                        dictionary = replaceDict.get(col) if not replaceDict.get(col) == None else {}
                        for thing in things:
                            dictionary.update({thing: 0})
                        for num in numbers:
                            dictionary.update({num: float(num.replace(',', ''))})
                        replaceDict.update({col:  dictionary.copy()})
                    else:
                        dictionary = replaceDict.get(col) if not replaceDict.get(col) == None else {}
                        for thing in data1[col].unique():
                            dictionary.update({thing: len(dictionary)})
                        replaceDict.update({col:  dictionary.copy()})
                data1 = load_data(path, startrows=rowsForSingleColumn * p, endrows=rowsForSingleColumn * (1 + p),
                                 cols=[i])
                p += 1
    return timeCols,replaceDict,colNums





def augment_data(data,timeCols,replaceDict,lastLog):
    #todo: try PCA

    # augment_data(data,timeCols,replaceDict )-> augmented_data,lastLog)
    # data – data we work with currently
    # timeCols - all columns with timestamps
    # replaceDict - list of columns and dictionaries to replace their values with
    #lastLog - dictionary of the last timestamp an id had
    # augmented_data – data, with timeCols replaced with the intervals between them, columns that are in the replaceDict are replaced with the integer for them
    for col in replaceDict:
        data=data.replace(replaceDict[col])

    if len(timeCols)==0:
        return data, lastLog
    for j in range(data.axes[0].size):
        id = data.iloc[j,0]
        if lastLog.get(id)==None:
            lastLog.update({id:0})
        temp = lastLog.get(id)
        lastLog.update({id: data[timeCols[0]][j]})
        for col in timeCols:
            temp2 = data[col][j].value - temp
            temp =data[col][j].value
            data[col][j]=temp2

    return data,lastLog

def getMedoidsModel(data,K):
    # getKmedioidsModel(data,K) -> model
    # data – same as above
    # K – number of groups we expect
    # model – the model for Kmedioids

    # i realized there ia a problem, i cannot auarantee all
    # cathegories will be in the dictionray
    # nor can i guarantee all points (used in KMedioids) will be in the first interation
    # therefore two options: use first iteration and pray all groups are represented
    # or find all mediod points for each iteration, then find the mediod for each group
    # it can be done though with much smaller groups
    maxIterations=30
    model = KMedoids(K,max_iter=maxIterations,init='k-medoids++')
    model.fit(data)
    return model

def predictGrouping (data,model):
    # predictGrouping (data,model) ->  dataGrouping
    # data – same save above
    # model – same as above
    # dataGrouping – data contacted with the grouping of model predictions on the data
    dataGrouping = model.predict(data)
    return dataGrouping




#_____________________________
#main just to check if the funcions work properly
#______________________________

def main(args):
    path = args.data_path
    assert os.path.isfile(path), "no file"
    assert path[-5:] == ".xlsx" or path[-4:] == ".xls" or path[-4:] == ".csv", "not excel or csv"
    try:
        row_num = decide_row_num_halt(path)
    except:
        print("failed to open file")
        return
    # data = load_data(path,endrows=row_num)
    t = time.time()
    timeCol, replaceDict, colNum = get_time_replace_reduct_cols(path, row_num)
    print("dictionaries: " + str(time.time() - t))

    row_num = decide_row_num_halt(path, colNum)
    lastLog = {}
    K = round(args.clusters_num)
    clusterPoints = pd.DataFrame()
    i = 0
    whileFlag = True
    data = load_data(path, row_num * (i), row_num * (i + 1))
    t = time.time()

    while not data.empty or whileFlag:
        data, lastLog = augment_data(data, timeCol, replaceDict, lastLog)
        model = getMedoidsModel(data, K)
        clusterPoints = pd.concat([clusterPoints, data.iloc[model.medoid_indices_]])
        i += 1
        data = load_data(path, row_num * (i), row_num * (i + 1), cols=colNum)
        whileFlag = False
    print("model training: " + str(time.time() - t))
    lastLog = {}
    model = getMedoidsModel(clusterPoints, K)
    i = 0
    data = load_data(path, row_num * (i), row_num * (i + 1))
    t = time.time()

    while not data.empty:
        data, lastLog = augment_data(data, timeCol, replaceDict, lastLog)
        prediction = predictGrouping(data, model)
        save_file(path, prediction, row_num * (i), row_num * (i + 1))
        data = load_data(path, row_num * (i + 1), row_num * (i + 2))
        i += 1
    print("save: " + str(time.time() - t))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Kmedioids')
    parser.add_argument('--data_path', default=None, type=str, help='path to data')
    parser.add_argument('--clusters_num', default=1, type=int, help='number of groupings')
    args = parser.parse_args()
    main(args)

