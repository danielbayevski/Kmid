

import os.path
# import time

import numpy
import pandas as pd
import sys
import psutil
import math
from sklearn_extra.cluster import KMedoids
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def load_data_xls(path, startRows = None,
                  endRows = None, cols = None, 
                  sheet = 0):
    header = pd.read_excel(path, usecols = cols, nrows = 1, sheet_name = sheet)
    temp = pd.read_excel(path,
                         usecols = cols,
                         nrows = endRows - startRows if
                         startRows else endRows, skiprows = startRows,
                         sheet_name = sheet)
    temp.columns = header.columns
    return temp


def load_data_csv(path, startRows = None, endRows = None, cols = None):
    header = pd.read_csv(path, usecols = cols, nrows = 1)
    temp = pd.read_csv(path,
                       usecols = cols,
                       nrows = endRows - startRows if
                       startRows else endRows,
                       skiprows = startRows,
                       thousands = ",")
    temp.columns = header.columns
    return temp


def load_data_type(path,
                   startRows = None,
                   endRows = None,
                   cols = None,
                   sheet = 0):
    if path[-5:] == ".xlsx" or path[-4:] == ".xls":
        return load_data_xls(path, startRows, endRows, cols, sheet)
    elif path[-4:] == ".csv":
        return load_data_csv(path, startRows, endRows, cols)


def load_data(path, startRows = None, endRows = None, cols = None, sheet = 0):
    # load_data(path, startRows, endRow, cols) -> data
    # path – path to file startRows – first row to load
    # endRow – last row to load, or the last of file, the minimum
    # cols – columns to load, there may be columns we would not want to load
    # as they are useless
    # data – return the loaded partial file
    try:
        return load_data_type(path, startRows, endRows, cols, sheet)
    except:
        return pd.DataFrame()


def save_file_xls2(new_path, temp, sheet):
    if not os.path.isfile(new_path):
        pd.DataFrame().to_excel(new_path, sheet_name = sheet)
    with pd.ExcelWriter(new_path, mode = 'a', if_sheet_exists = 'replace') \
            as writer:
        temp.to_excel(writer, sheet_name = sheet)


def save_file_xls(path, temp, sheet, newDirectory):
    new_path = newDirectory+path[-(len(path)-path.rfind("/")):]
    save_file_xls2(new_path, temp, sheet)


def save_file_csv(path, temp, newDirectory):
    new_path = newDirectory+path[-(len(path)-path.rfind("/")):]
    temp.to_csv(new_path, mode = 'a')


def save_file_type(path, temp, newDirectory, sheet = None):
    if path[-5:] == ".xlsx" or path[-4:] == ".xls":
        save_file_xls(path, temp, sheet, newDirectory)
    elif path[-4:] == ".csv":
        save_file_csv(path, temp, newDirectory)


def save_file(path, prediction, newDirectory, startRows = None,
              endRows = None, sheet = 'Sheet1'):
    # save_data(path, startRows, endRow)
    # path – path to the file to load to, 
    #               or create a blank file, titled “path_new.csv/xls”
    # startRows – first row to save
    # endRow – last row to load, or the last of file, the minimum
    temp = load_data(path, startRows, endRows, sheet = sheet)
    temp["grouping"] = prediction.tolist()
    save_file_type(path, temp, newDirectory, sheet)


def decide_row_num_halt(path, colNums = None):
    # decide how many rows can you work with at a time, depends on the
    # dataBuffer
    dataBuffer = psutil.virtual_memory().free / 2
    temp = load_data(path, 0, 1, colNums)
    return round(dataBuffer / sys.getsizeof(temp))


def temp_exchange(data, temp):
    temp2 = data.value - temp
    temp = data.value
    return temp2, temp


def last_log_update(data, lastLog, j, timeCols, id_number):
    if not lastLog.get(id_number):
        lastLog.update({id_number: 0})
    temp = lastLog.get(id_number)
    lastLog.update({id_number: data[timeCols[0]][j]})
    return temp, lastLog


def time_augment2(data, timeCols, lastLog, j):
    temp, lastLog = last_log_update(data, lastLog, j, timeCols, 
                                    data.iloc[j, 0])
    for col in timeCols:
        data[col][j], temp = temp_exchange(data[col][j], temp)
    return data, lastLog


def time_augment(data, timeCols, lastLog):
    for j in range(data.axes[0].size):
        data, lastLog = time_augment2(data, timeCols, lastLog, j)
    return data, lastLog


def getMedoidsModel(data, K):
    # getKmedioidsModel(data, K) -> model
    # data – same as above
    # K – number of groups we expect
    # model – the model for Kmedioids

    # i realized there ia a problem, i cannot guarantee all
    # categories will be in the dictionary
    # nor can i guarantee all points (used in KMedioids) will be in the 
    # first iteration therefore two options: use first iteration and pray all 
    # groups are represented or find all mediod points for each iteration, 
    # then find the mediod for each group it can be done though with much 
    # smaller groups
    maxIterations = 40
    model = KMedoids(K, max_iter = maxIterations, init = 'k-medoids++')
    model.fit(data)
    return model


def standardize(data, object_number_cols, real_number_cols, mean_std):
    data = numberize(data, object_number_cols, list(object_number_cols) +
                     list(real_number_cols))
    for i in list(object_number_cols) + list(real_number_cols):
        data[i] = (data[i] - mean_std[i][0])/mean_std[i][1]
    return data


def numberize(data, object_cols, real_and_object_cols):
    for i in object_cols:
        data[i] = pd.to_numeric(data[i].map(lambda x: x.replace(", ", "")), 
                                errors = 'coerce)')
    data[real_and_object_cols] = data[real_and_object_cols].fillna(
            data[real_and_object_cols].mean())
    return data


def one_hot_encoding2(data, i, labels_dictionary, zero_array, values, tilt):
    for k, j in enumerate(data[i]):  # there is a better way to do it probably
        zero_array[k, labels_dictionary[i][str(j)]] = 1
    return (numpy.hstack((zero_array,
                          values[:, :data.columns.get_loc(i) + tilt],
                          values[:, data.columns.get_loc(i) + tilt + 1:])),
            tilt + len(labels_dictionary[i]) - 1)


def one_hot_encoding(data, columns, labels_dictionary):
    # # dt = numpy.zeros((data.shape[0], 1))
    values, tilt = data.values, 0
    for i in columns:
        values, tilt = one_hot_encoding2(data,
                                         i,
                                         labels_dictionary,
                                         numpy.zeros((
                                             data[i].shape[0],
                                             len(labels_dictionary[i]))),
                                         values, tilt)
    #     # df = numpy.zeros((data[i].shape[0], len(labels_dictionary[i])))
    #     # for k, j in enumerate(data[i]):
    #     #     df[k, labels_dictionary[i][str(j)]]=1
    #     # le, loc = len(labels_dictionary[i]), data.columns.get_loc(i)+tilt
    #     # vals = numpy.hstack((df, vals[:, :loc], vals[:, loc+1:]))
    #     #tilt += le - 1
    return values

    # lst = [numpy.where(data.columns.values == columns[j])[0][0]
    # for j in range(columns.shape[0])]
    # ct = ColumnTransformer(transformers =
    #                         [('encoder', OneHotEncoder(sparse = False),
    #                           lst)], remainder = "passthrough")
    # return ct.fit_transform(data)


def label_encoding(data, columns):
    lc = LabelEncoder()
    # d = lc.fit_transform(data[i])
    # d=data.copy()
    for i in columns:
        data[i] = lc.fit_transform(data[i].astype(str))
        # data[i] = label_dictionary[i].transform(data[i].astype(str))
    return data


def augment(data, label_s_cols, label_l_cols,
            object_number_cols, real_number_cols, 
            time_cols, mean_std, lastLog, labels_dictionary):
    data = standardize(data, object_number_cols, real_number_cols, mean_std)
    data, lastLog = data, lastLog if len(time_cols) == 0 else \
        time_augment(data, time_cols, lastLog)
    data = label_encoding(data, label_l_cols)
    # , labels_dictionary)
    return one_hot_encoding(data, label_s_cols, labels_dictionary), lastLog


def predictFile(path, K, sheet = 0, newDirectory = None):
    # "main" function
    newDirectory = newDirectory if newDirectory else path
    if not os.path.isfile(path):
        raise FileNotFoundError("no file")
    if path[-5:] != ".xlsx" and path[-4:] != ".xls" \
            and path[-4:] != ".csv":
        raise TypeError("inappropriate type of file, use CSV or"
                        " XLS/XLSX files only")

    try:
        row_num = decide_row_num_halt(path)
    except:
        print("failed to open file")
        return

    # note: i feel like this could have been a function, to lessen the clutter
    # data – data we work with currently
    # timeCols – all columns with timestamps
    # replaceDict  - list of columns and dictionaries to replace
    #   their values with
    # colNums – list of columns to load, as some of them are useless, example:
    # personal numbers, passwords and OTP

    data = load_data(path, endRows = row_num, sheet = sheet)
    time_cols = []
    numberTypes = numpy.array(
            ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint64", 
             "uint32", 
             "float32", "float64"])
    label_s_cols = []
    label_l_cols = []
    object_number_cols = []
    real_number_cols = []

    for i, col in enumerate(data.columns):
        if data[col].dtype.name[:8] == "datetime":
            time_cols.append(col)
        elif data[col].dtype.name in numberTypes:
            if not data[col].is_monotonic_increasing:
                real_number_cols.append(i)
        else:
            if 1 < data[col].nunique() < 0.5*row_num:
                try:
                    for val in data[col].values[:100]:
                        val = val.replace(", ", "")
                        float(val)
                    object_number_cols.append(i)
                except:
                    label_l_cols.append(i)

    colNum = label_l_cols + time_cols + real_number_cols + object_number_cols
    colNum.sort()
    labels_dictionary = {}
    label_l_cols_temp = label_l_cols.copy()

    for i in label_l_cols_temp:
        data = load_data(path, endRows = row_num, cols=[i], sheet = sheet)
        val_set = []
        p = 0
        column_name = data.columns[0]

        while not data.empty:
            val_set = val_set + \
                      data[data.columns[0]].unique().astype(str).tolist()
            p += 1
            data = load_data(path, row_num*p,
                             row_num*(p+1), cols = [i], sheet = sheet)

        val_set.sort()
        val_set = set(val_set)
        if len(val_set) < 5:
            label_l_cols.remove(i)
            label_s_cols.append(i)
            labels_dictionary.update(
                    {column_name: {element: j for j, element in
                                   enumerate(val_set)}})
        else:
            labels_dictionary.update(
                    {column_name: LabelEncoder().fit(list(val_set))})

    single_col_row_num = decide_row_num_halt(path, [real_number_cols[0]])
    mean_std = {}

    for i in real_number_cols:
        data = load_data(path, endRows = single_col_row_num,
                         cols = [i], sheet = sheet)
        mean_std.update(
                {data.columns[0]: list(data.describe().values[1:3, 0])})

    if len(object_number_cols) > 0:
        single_col_row_num = decide_row_num_halt(path, [object_number_cols[0]])

        for i in object_number_cols:
            data = load_data(path, endRows = single_col_row_num, cols = [i], 
                             sheet = sheet)
            data = pd.to_numeric(
                data.map(lambda x: x.replace(", ", "")), errors = 'coerce')
            mean_std.update({data.columns[0], data.describe().values[1:3, 0]})

    data = load_data(path, endRows = 1, sheet = sheet)
    label_l_cols = data.columns[label_l_cols]
    label_s_cols = data.columns[label_s_cols]
    object_number_cols = data.columns[object_number_cols]
    real_number_cols = data.columns[real_number_cols]
    time_cols = data.columns[time_cols]
    row_num = decide_row_num_halt(path, colNum)
    lastLog = {}
    clusterPoints = pd.DataFrame()
    i, rowNumSqrt = 0, round(math.sqrt(row_num))
    data = load_data(path, None, rowNumSqrt, cols = colNum, sheet = sheet)

    while not data.empty:
        data, lastLog = augment(data, label_s_cols, label_l_cols, 
                                object_number_cols, real_number_cols,
                                time_cols, mean_std, lastLog,
                                labels_dictionary)
        data = numpy.nan_to_num(data)
        if data.shape[0] >= K:
            model = getMedoidsModel(data, K)
            clusterPoints = pd.concat([clusterPoints, 
                                       pd.DataFrame(
                                               data[model.medoid_indices_])])

        i += 1
        data = load_data(path, rowNumSqrt*i, rowNumSqrt*(i+1), cols = colNum,
                         sheet = sheet)

    lastLog, i = {}, 0
    model = getMedoidsModel(clusterPoints, K)
    data = load_data(path, row_num * i, row_num * (i + 1), colNum, sheet)
    new_sheet = sheet if sheet != 0 else "Sheet1"
    if os.path.isfile(newDirectory+path[-(len(path)-path.rfind("/")):]):
        os.remove(newDirectory+path[-(len(path)-path.rfind("/")):])

    while not data.empty:
        data, lastLog = augment(data, label_s_cols, label_l_cols, 
                                object_number_cols, real_number_cols,
                                time_cols, mean_std, lastLog,
                                labels_dictionary)
        data = numpy.nan_to_num(data)
        prediction = model.predict(data)
        save_file(path, prediction, newDirectory,
                  row_num * i, row_num * (i + 1),
                  sheet = new_sheet)
        i += 1
        data = load_data(path, row_num * i, row_num * (i + 1),
                         cols = colNum, sheet = sheet)
