

import argparse
import os.path
import time

import numpy
from sklearn_extra.cluster import KMedoids
import pandas as pd
import sys
import psutil
import math
DATABUFFER = math.sqrt(psutil.virtual_memory().free) # 8 MB
# import datetime



def load_data(path,startrows=None,endrows=None,cols=None):
    # load_data(path, startRow, endRow, cols) -> data
    # path – path to file startRow – first row to load
    # endRow – last row to load, or the last of file, the minimum
    # cols – columns to load, there may be columns we would not want to load as they are useless
    # data – return the loaded partial file
    row_num = endrows-startrows if startrows else endrows
    if path[-5:] == ".xlsx" or path[-4:] == ".xls":
        temp = pd.read_excel(path,usecols=cols,nrows=row_num,skiprows=startrows)
    elif path[-4:] == ".csv":
        temp=pd.read_csv(path,usecols=cols,nrows=row_num,skiprows=startrows)
    return temp

def save_file(path, pred, startrows=None, endrows=None):
    # save_data(path, startRow, endRow)
    # path – path to the file to load to,
    #               or create a blank file, titled “path_new.csv/xls”
    # startRow – first row to save
    # endRow – last row to load, or the last of file, the minimum
    # TODO:find better way to partially save file,
    #  dont want to take too much memory
    if path[-5:] == ".xlsx":
        temp = pd.read_excel(path, nrows=endrows - startrows, skiprows=startrows)
        new_path = path[:-5] + "_new" + path[-5:]
        temp["grouping"] = pred.tolist()
        temp.to_excel(new_path)
    elif path[-4:] == ".xls":
        temp = pd.read_excel(path, nrows=endrows - startrows, skiprows=startrows)
        new_path = path[:-4] + "_new" + path[-4:]
        temp["grouping"] = pred.tolist()
        temp.to_excel(new_path)
    elif path[-4:] == ".csv":
        temp = pd.read_csv(path, nrows=endrows - startrows, skiprows=startrows)
        new_path = path[:-4] + "_new" + path[-4:]
        temp["grouping"] = pred.tolist()
        temp.to_csv(new_path)


def decide_row_num_halt(path,cols=None):
    #decide how many rows can you work with at atime, depends on the DATABUFFER
    if path[-5:] == ".xlsx" or path[-4:] == ".xls":
        temp = pd.read_excel(path,usecols=cols,nrows=1)
    elif path[-4:] == ".csv":
        temp=pd.read_csv(path,usecols=cols,nrows=1)
    return round(DATABUFFER/ sys.getsizeof(temp))


def get_time_replace_reduct_cols(data):
    #get_time_replace_reduct_cols(data) -> timeCols,replaceDict,colNums
    # data – data we work with currently
    # timeCols – all columns with timestamps
    # replaceDict  - list of columns and dictionaries to replace their values with
    # colNums – list of columns to load, as some of them are useless, example:  personal numbers, passwords and OTP
    timeCols= []
    replaceDict = []
    colNums = []
    row = data.iloc[0]
    colLen = len(data.axes[1])
    for i,col in enumerate(data.columns):
        if data[col].dtype.name[:8]=="datetime":
            timeCols.append(col)
            colNums.append(i)

        #check for OTP
        elif isinstance(row[col],str) :

            things =[]
            for j in data[col]:
                if not (j in things):
                    if len(things)>= colLen:
                        break
                    things.append(j)
            if len(things) < colLen :

                dictionay={}
                for j, thing in enumerate(things):
                    dictionay.update({thing:j})
                replaceDict.append((col,dictionay.copy()))
                colNums.append(i)
        else:
            colNums.append(i)

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
        data=data.replace({col[0]:col[1]})

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
    maxIterations=10
    model = KMedoids(K,max_iter=maxIterations,init='k-medoids++')
    t=time.time()
    model.fit(data)
    print(str(time.time()-t))
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
    assert os.path.isfile(path),"no file"
    assert path[-5:] == ".xlsx" or path[-4:] == ".xls" or path[-4:] == ".csv", "not excel or csv"
    try:
        row_num = decide_row_num_halt(path)
    except:
        print("failed to open file")
        return
    data = load_data(path,endrows=row_num)
    timeCol,replaceDict,colNum = get_time_replace_reduct_cols(data)
    row_num = decide_row_num_halt(path,colNum)
    data = load_data(path, endrows=row_num,cols=colNum)
    lastLog={}
    data,lastLog = augment_data(data,timeCol,replaceDict,lastLog)
    K=round(args.clusters_num)
    model = getMedoidsModel(data,K)
    i=0
    while not data.empty:
        prediction = predictGrouping(data,model)
        save_file(path, prediction,row_num*(i),row_num*(i+1))
        data = load_data(path,row_num*(i+1),row_num*(i+2))
        data,lastLog = augment_data(data,timeCol,replaceDict,lastLog)

    #
    #
    #
    # i=1
    # while True:
    #     data=augment_data(load_data(path,row_num*(i+1),row_num*i,skip_columns))
    #     if (data == None):
    #         break
    #     prediction = model.predict(data)
    #     save_file(path, prediction, row_num * (i))







if __name__ == '__main__':
    parser = argparse.ArgumentParser('Kmedioids')
    parser.add_argument('--data_path', default=None, type=str, help='path to data')
    parser.add_argument('--clusters_num', default=1, type=int, help='number of groupings')
    args = parser.parse_args()
    main(args)

