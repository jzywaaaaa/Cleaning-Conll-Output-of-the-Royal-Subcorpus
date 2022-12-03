# author: Ziyue Wang
# date: 2022 December 2


import pandas as pd
import re
import numpy as np


####### get files #######
import sys

subcorpus_file_loc = './royal_subcorpus_data/' + sys.argv[1] 
atf_converted_file_loc = './output/' + sys.argv[1]

print("Subcorpus_file: ", subcorpus_file_loc, "atf_converted_file_loc: ", atf_converted_file_loc)

# store the text number (e.g., P216736) for later use
text_number = re.findall(pattern = '.*\/([^\/]*)$', string = subcorpus_file_loc)[0][0:-6]


######## define tools ########

# read_conll takes in a subcorpus conll file and outputs a dataframe
def read_conll(filepath):
    with open(filepath) as fp:
        idx = 0
        while True:
            l = fp.readline()
            if l[0] != "#":
                break
            idx += 1
        print(idx, "lines starting with #")
        print(fp.tell())
        # read in the conll file
        data = pd.read_fwf(fp)
    # extract useful information from the conll file
    useful_info = data.iloc[:,0].str.split('\t')
    # prepare a clean dataframe
    cleaned_data = pd.DataFrame({'ID': [], 'WORD': [], 'SEGM': [], 'POS': [], \
        'MORPH': [], 'HEAD': [], 'EDGE': [], 'MISC': []})
    # update the clean dataframe
    for i in range(cleaned_data.shape[1] - 1):
        cleaned_data.iloc[:, i] = [row[i] for row in useful_info]
    cleaned_data.iloc[:, -1] = [row[-1] for row in useful_info]

    return cleaned_data


# extract_id takes an ATF converted file and extracts the ID column as a vector
def extract_id(filepath):
    # read in the conll file converted by the ATF converter
    data = pd.read_fwf(filepath)
    useful_info = data.iloc[1:, 0].str.split("\t")
    # get only the id
    id_col = [row[0] for row in useful_info]
    # clean the id (want: o.1.1, but have: s1.1.1, o.col1.1.1)
    for id in id_col:
        ans = id[0] # get only the first character
        numberings = re.findall(pattern = "[0-9]", string = id) # get all numbers in the string
        for num in numberings:
            ans = ans + "." + num
        id = ans
    return id_col


######## perform the clean up #######

df = read_conll(subcorpus_file_loc)

# rename the WORD and MORPH column
df = df.rename(columns={'WORD': 'FORM', 'MORPH': 'XPOSTAG'})
# drop the EDGE and POS columns
df = df.drop(columns = ['EDGE', 'POS'])
# add the DEPREL column
df['DEPREL'] = np.repeat('_', df.shape[0])
# update the ID column
df['ID'] = extract_id(atf_converted_file_loc)
# reorder the columns
df = df[['ID', 'FORM', 'SEGM', 'XPOSTAG', 'HEAD', 'DEPREL', 'MISC']]



####### write the cleaned df to a conll file ########

with open(text_number + ".conll", "w") as fp:
    fp.write("#new_text=" + text_number + "\n")

df.to_csv(text_number + '.conll', header = True, index = None, sep = '\t', mode = 'a')