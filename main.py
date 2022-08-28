

import argparse
from sklearn_extra.cluster import KMedoids
import pandas as pd
import sys
DATABUFFER = 268435456 # size of quarter of a gigabyte




def load_data(path,startrows=None,endrows=None,cols=None):
    # load_data(path, startRow, endRow, cols) -> data
    # path – path to file startRow – first row to load
    # endRow – last row to load, or the last of file, the minimum
    # cols – columns to load, there may be columns we would not want to load as they are useless
    # data – return the loaded partial file
    row_num = endrows-startrows if startrows else None
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
        temp = pd.concat(temp, pred)
        temp.to_excel(new_path)
    elif path[-4:] == ".xls":
        temp = pd.read_excel(path, nrows=endrows - startrows, skiprows=startrows)
        new_path = path[:-4] + "_new" + path[-4:]
        temp = pd.concat(temp, pred)
        temp.to_excel(new_path)
    elif path[-4:] == ".csv":
        temp = pd.read_csv(path, nrows=endrows - startrows, skiprows=startrows)
        new_path = path[:-4] + "_new" + path[-4:]
        temp = pd.concat(temp, pred)
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
    colLen = len(data)
    for i,col in enumerate(data.columns):
        #check for OTP
        if isinstance(row[col],str) :
            #TODO:add time check
            things =[]
            for j in data[col]:
                if not (j in things):
                    if len(things)>= colLen/2:
                        break
                    things.append(j)
            if len(things) < colLen / 2:
                colNums.append(i)
                dictionay={}
                for j, thing in enumerate(things):
                    dictionay.update({thing:j})
                replaceDict.append((col,dictionay.copy()))
                colNums.append(i)
        else:
            colNums.append(i)

    return timeCols,replaceDict,colNums


def augment_data(data): #todo: aguament data then PCA it to death

    pass



#_____________________________
#main just to check if the funcions work properly
#______________________________


def main(args):
    # TODO: add assert args.data_path is a real file with ending csv,xls xslx, and that it can be loaded and read
    path = args.data_path
    row_num = decide_row_num_halt(path)
    data = load_data(path,endrows=row_num)
    timeCol,replaceDict,colNum = get_time_replace_reduct_cols(data)
    row_num = decide_row_num_halt(path,colNum)
    # K=args.clusters_num
    #
    # model = KMedoids(K)
    # data=load_data(path, row_num,cols=skip_columns)
    # data = augment_data(data)
    # model.fit(data)
    # prediction = model.predict(data)
    #
    # save_file(path, prediction)
    #
    # i=1
    # while True:
    #     data=augment_data(load_data(path,row_num*(i+1),row_num*i,skip_columns))
    #     if (data == None):
    #         break
    #     prediction = model.predict(data)
    #     save_file(path, prediction, row_num * (i))
    pass






if __name__ == '__main__':
    parser = argparse.ArgumentParser('Kmedioids')
    parser.add_argument('--data_path', default=None, type=str, help='path to data')
    parser.add_argument('--clusters_num', default=1, type=int, help='number of groupings')
    args = parser.parse_args()
    main(args)

