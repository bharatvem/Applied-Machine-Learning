'''
Created on Sep 14, 2016

@author: Bharat Vemulapalli
'''
import pandas as pd
import math
from click._compat import raw_input


###Read the file monks-1.train
df=pd.read_table("monks-1.train",header=None,sep=" ")
df=df.drop(0,1) #remove the first NaN column
df.columns=['class','a1','a2','a3','a4','a5','a6','id'] #name the columns
columns=['a1','a2','a3','a4','a5','a6']
col_types={'a1':'cat','a2':'cat','a3':'cat','a4':'cat','a5':'cat','a6':'cat'} #Column types
nodes = pd.DataFrame(columns=['node','child1','child2','col','value','leaf','class'])





###############-------FUNCTIONS REQUIRED FOR DECISION TREE-----------####################
def calc_entropy(a):
    counts = pd.value_counts(a['class'].values,sort=False)
#     print('counts'+':'+str(len(a.index)))
    entropy = 0
    for i in a['class'].unique():
        p = counts[i]/len(a.index)
        entropy += p*math.log2(p)
    return -entropy
    
def splitrow(dataframe,col,value):
    global col_types
    if(col_types[col]=='num'):
        left = dataframe[dataframe[col]<=value]
        right = dataframe[dataframe[col]>value]
    else:
        left = dataframe[dataframe[col]==value]
        right = dataframe[dataframe[col]!=value]
    return left,right
        
def get_split_criterion(b):
    high_gain_colsplit = 0
    col_to_split = ''
    value_to_split,value_to_split_col = 0,0
#     print('inget_split_criterion')
    for i in columns:
        parent_e = calc_entropy(b)
        parent_size = len(b.index)
        
        high_gain_valsplit = 0
        for j in b[i].unique(): # for every unique class in the column
            l,r=splitrow(b, i, j)
            child_avg_e = ((len(l.index)/parent_size)*calc_entropy(l))+((len(r.index)/parent_size)*calc_entropy(r))
            #         print(child_avg_e)
            info_gain = parent_e - child_avg_e
            if(info_gain >= high_gain_valsplit):
                high_gain_valsplit = info_gain
                value_to_split_col = j
        if(high_gain_valsplit >= high_gain_colsplit):
                high_gain_colsplit = high_gain_valsplit
                col_to_split = i
                value_to_split = value_to_split_col
    
    return col_to_split,value_to_split

def split_model_create(data,nn,d):
    global nodes
    if(d>0 and (len(data.index)>1) and (len(data['class'].unique())>1)):
#         print('in if')
        split_col,split_val = get_split_criterion(data)
        ldata,rdata = splitrow(data, split_col, split_val)
        tempr = pd.DataFrame([[nn,2*nn,(2*nn)+1,split_col,split_val,0,'']],columns=['node','child1','child2','col','value','leaf','class'])
        nodes=nodes.append(tempr,ignore_index=True)
#         d=d-1
        split_model_create(ldata, 2*nn, d-1)
        split_model_create(rdata, (2*nn)+1, d-1)
    else:
#         print('in else')
#         print(data['class'].value_counts())
        tempr = pd.DataFrame([[nn,0,0,'','',1,data['class'].value_counts().idxmax()]],columns=['node','child1','child2','col','value','leaf','class'])
        nodes=nodes.append(tempr,ignore_index=True)

def predictor():
    print('pr')

###############-------END OF FUNCTIONS REQUIRED FOR DECISION TREE-----------####################

### The actual start of the processing to create the model
depth = int(raw_input("Enter depth of the tree:"))
# print(df)
split_model_create(df, 1, depth)
# print(pd.value_counts(df['class'].values,sort=True))
# print(df['class'].value_counts().idxmax())
# print(df.mode('class'))
print(nodes)
###Read the file monks-1.test
# dftest = pd.read_table("monks-1.test",header=None,sep=" ")
# # dftest=dftest.drop(0,1) #remove the first NaN column
# dftest.columns=['predicted','class','a1','a2','a3','a4','a5','a6','id'] #name the columns
# # dftest['predicted']='x'
# for index,row in dftest.iterrows():
#     print(str(index)+str(row['class']))
#     if index==50:
#         break;
# print(dftest)