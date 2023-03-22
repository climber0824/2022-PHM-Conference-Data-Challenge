import pandas as pd
import glob

def data_preprocessing(path):
    # combine data_po files into df1
    files_po = glob.glob(path + "*data_po*.csv")    
    df1 = pd.DataFrame()
    for file in files_po:
        with open(file, 'r') as f:
            csv_string = f.read()
            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)
            temp_df = df.iloc[:, :500]
            df1 = pd.concat([df1, temp_df], axis=0)

    # rename the columns in df1 and convert to float
    df1.columns = [i for i in range(df1.shape[1])]
    df1 = df1.rename(columns={0: 'gt'})
    df1 = df1.astype(float)

    # combine data_pdmp files into df2
    files_pdmp = glob.glob(path + "*data_pdmp*.csv")
    df2 = pd.DataFrame()
    for file in files_pdmp:
        with open(file, 'r') as f:
            csv_string = f.read()
            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)
            temp_df = df.iloc[:, :500]
            df2 = pd.concat([df2, temp_df], axis=0)

    # rename the columns in df2 and convert to float
    df2.columns = [i for i in range(df2.shape[1])]
    df2 = df2.rename(columns={0: 'gt'})
    df2 = df2.astype(float)

    # combine data_pin files into df3
    files_pin = glob.glob(path + "*data_pin*.csv")
    df3 = pd.DataFrame()
    for file in files_pin:
        with open(file, 'r') as f:
            csv_string = f.read()
            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)
            temp_df = df.iloc[:, :500]
            df3 = pd.concat([df3, temp_df], axis=0)

    # rename the columns in df3 and convert to float
    df3.columns = [i for i in range(df3.shape[1])]
    df3 = df3.rename(columns={0: 'gt'})
    df3 = df3.astype(float)

    return df1, df2, df3