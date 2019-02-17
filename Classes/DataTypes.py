import numpy                 as np
import pandas                as pd
from   sklearn.externals import joblib
from   datetime          import datetime  


###########################################
# Parameters for the classification study #
###########################################
class binary_params:
    def __init_(self,notes):
        self.start_time = datetime.now()
        self.notes      = notes
        #self. 

    def set_xs(self,xs):
        self.xs         = xs
  
    def set_sgn_name(self,name):
        self.sgn_name   = name
   
    def set_bkg_name(self,name):
        self.bkg_name   = name

    def set_sgn_mass_list(self,lst):
        self.sgn_mass_list = lst
  
    def set_sgn_ctau_list(self,lst):
        self.sgn_ctau_list = lst

    def set_train_val_test_ratio(self,tvtr):
        self.train_val_test_ratio = tvtr






##########################################
# pkl files that store pandas DataFrames #
##########################################
class pkl_df:
    def __init__(self,pth,fn): #(self,pth,fn,data_typ)
        self.name    = fn
        self.path    = pth
        #self.data_typ = data_typ
        #if self.data_typ == 'h5':
        #    store        = pd.HDFStore(self.path + '/' +self.name)
        #    self.df      = store.select('table')
        #elif self.data_typ = 'pkl':  
        if   '.h5'  in self.name:
            self.df      = pd.read_hdf(self.path + '/' +self.name, 'df') #'df' is the key
        elif '.pkl' in self.name:
            self.df      = joblib.load( self.path + '/' +self.name )  
        
        #self.df      = (self.df).dropna()  
        
        firstColName = (self.df).columns.values[0]
        dropone      = (self.df)[firstColName] != -1
        self.df      = (self.df)[dropone]
        self.events  = len(self.df)
        self.train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}

    def set_xs(self,xs):
        self.xs       = xs
        self.df['xs'] = xs

    def set_label(self,label,label_str):
        self.label                = label
        self.df[label_str]        = label

    # Split the DataFrame into train/val/test sections:
    def split(self,train_val_test_ratio):
        self.train_val_test_ratio = train_val_test_ratio
        self.n_train              = int( self.events * train_val_test_ratio['train'] )
        self.n_val                = int( self.events * train_val_test_ratio['val']   )
        self.n_test               = self.events - self.n_train - self.n_val 
        self.df_train             = (self.df).iloc[:self.n_train] 
        self.df_val               = (self.df).iloc[self.n_train:self.n_train+self.n_val]
        self.df_test              = (self.df).iloc[self.n_train+self.n_val:]        
        df_dict = {'train': self.df_train, 'val': self.df_val, 'test': self.df_test}
        return df_dict

    def read_n(self,tvt_str):
        if   tvt_str == 'train': return self.n_train
        elif tvt_str == 'val'  : return self.n_val
        elif tvt_str == 'test' : return self.n_test
    
    # Shift column names: for example 'E_0' to 'E_1'
    def shift_col_num(self,sh_n):
        self.old_col_list = list(self.df.columns.values)
        tmp_list = []
        for str_i in self.old_col_list:
            pos_i   = str_i.find('_')
            pre_str = str_i[:pos_i]
            num_str = int( str_i[pos_i+1:] ) 
            new_str = pre_str+'_'+str( num_str + sh_n ) 
            tmp_list.append(new_str)
        self.new_col_list = tmp_list
        self.df.columns = self.new_col_list

    def set_weight(self,weight,weight_str):
        self.df_train[weight_str] = weight['train']
        self.df_val[weight_str]   = weight['val']
        self.df_test[weight_str]  = weight['test']


    def splitN(self,n_fold):
        df_bkg_dic     = {}  
        k_fold_bkg_dic = {i:[] for i in range(n_fold)} 
        k_fold_bkg     = []         

        df         = self.df 
        df_bkg     = df[df['is_signal_new']==0]
        df_sgn     = df[df['is_signal_new']==1] 
        qcd_n_dic  = dict(df_bkg.groupby('xs').size() )        
       
        for key, item in qcd_n_dic.iteritems():
            print '#events for xs=', key, ': ', item 
            tmp_df = df_bkg[df_bkg['xs']==key]
            #batch_size = int(item / float(n_fold))
            df_bkg_dic[key] = np.array_split(tmp_df, n_fold)   
            for i in range(n_fold):    k_fold_bkg_dic[i].append( df_bkg_dic[key][i] )
                
        k_fold_sgn          = np.array_split(df_sgn, n_fold) 
        for i in range(n_fold):    k_fold_bkg.append( pd.concat(k_fold_bkg_dic[i]) )
        
        return k_fold_sgn, k_fold_bkg



























