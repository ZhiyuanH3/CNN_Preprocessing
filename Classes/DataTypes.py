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

        #self.df      =         
        
        firstColName = (self.df).columns.values[0]
        dropone      = (self.df)[firstColName] != -1
        self.df      = (self.df)[dropone]
 
        dropempty    = (self.df)['E_0'] != 0
        self.df      = (self.df)[dropempty]
        self.df      = (self.df)[ (self.df).notnull().any(1) ]        

        self.events  = len(self.df)
        self.train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}
 
     

        #self.jets    = (self.df).groupby('jetIndex_0') 

        #self.jets_no_match = (self.df).groupby('jetIndex_0')  
        #self.jets    = (self.df[self.df['isGenMatched']==1]).groupby('jetIndex_0')


    #def gen_matching(self):    self.df = self.df[ self.df['isGenMatched']==1 ]
    def gen_matching(self, gm):
        if gm:    self.jets    = (self.df[self.df['isGenMatched']==1]).groupby('jetIndex_0')
        else :    self.jets    = (self.df).groupby('jetIndex_0')  


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


    def splitTV_0(self,n_fold):
        df_bkg_dic_j0     = {}
        df_bkg_dic_j1     = {}
        k_fold_bkg_dic_j0 = {i:[] for i in range(n_fold)}
        k_fold_bkg_dic_j1 = {i:[] for i in range(n_fold)}
        k_fold_bkg_j0     = []
        k_fold_bkg_j1     = []

        df         = self.df
        df_bkg     = df[df['is_signal_new']==0]
        df_bkg_j0  = df_bkg[df_bkg['jetIndex_0']==0]
        df_bkg_j1  = df_bkg[df_bkg['jetIndex_0']==1]
        df_sgn     = df[df['is_signal_new']==1]

        qcd_n_dic_j0 = dict(df_bkg_j0.groupby('xs').size() )
        qcd_n_dic_j1 = dict(df_bkg_j1.groupby('xs').size() )  

        print '>>>> j0:'
        for key, item in qcd_n_dic_j0.iteritems():
            print '#events for xs=', key, ': ', item
            tmp_df = df_bkg_j0[df_bkg_j0['xs']==key]
          
            df_bkg_dic_j0[key] = np.array_split(tmp_df, n_fold)
            for i in range(n_fold):    k_fold_bkg_dic_j0[i].append( df_bkg_dic_j0[key][i] )

        print '>>>> j1:'
        for key, item in qcd_n_dic_j1.iteritems():
            print '#events for xs=', key, ': ', item
            tmp_df = df_bkg_j1[df_bkg_j1['xs']==key]
           
            df_bkg_dic_j1[key] = np.array_split(tmp_df, n_fold)
            for i in range(n_fold):    k_fold_bkg_dic_j1[i].append( df_bkg_dic_j1[key][i] )



        k_fold_sgn          = np.array_split(df_sgn, n_fold)
        for i in range(n_fold):    k_fold_bkg_j0.append( pd.concat(k_fold_bkg_dic_j0[i]) )
        for i in range(n_fold):    k_fold_bkg_j1.append( pd.concat(k_fold_bkg_dic_j1[i]) )

        return k_fold_sgn, k_fold_bkg_j0, k_fold_bkg_j1



    def splitTV_1(self, n_fold, upLmt=100000):
        #upLmt             = 100000 
        n_jets            = 2  
        df_bkg_dic        = {}
        k_fold_bkg_dic    = {}
        k_fold_bkg        = {} 
        for j in range(n_jets): 
            df_bkg_dic[j]     = {}
            k_fold_bkg_dic[j] = {i:[] for i in range(n_fold)}
            k_fold_bkg[j]     = []

        df         = self.df
        df_sgn     = df[df['is_signal_new']==1]
        df_bkg_t   = df[df['is_signal_new']==0]
        df_bkg     = {}
        qcd_n_dic  = {}
        for j in range(n_jets):
            df_bkg[j]  = df_bkg_t[df_bkg_t['jetIndex_0']==j] 
            qcd_n_dic[j] = dict(df_bkg[j].groupby('xs').size() )

        for j in range(n_jets):
            print '>>>> j'+str(j)+':'
            for key, item in qcd_n_dic[j].iteritems():
                print '#events for xs=', key, ': ', item
                #tmp_df             = df_bkg[j][df_bkg[j]['xs']==key]
                tmp_df             = (df_bkg[j][df_bkg[j]['xs']==key])[:upLmt]
                print '#events for xs=', key, '(with upper limit): ', len(tmp_df)   

                df_bkg_dic[j][key] = np.array_split(tmp_df, n_fold)
                for i in range(n_fold):    k_fold_bkg_dic[j][i].append( df_bkg_dic[j][key][i] )

        k_fold_sgn = np.array_split(df_sgn, n_fold)

        for j in range(n_jets):
            for i in range(n_fold):  
                k_fold_bkg[j].append( pd.concat(k_fold_bkg_dic[j][i]) )

        return k_fold_sgn, k_fold_bkg[0], k_fold_bkg[1]






    def splitTV(self, n_fold=6, upLmt=None, n_jets=4):
        out_dic           = {}
        df_bkg_dic        = {}
        #k_fold_bkg_dic    = {}
        #k_fold_bkg        = {} 
        k_fold_bkg        = []
        k_fold_bkg_dic    = {i:[] for i in range(n_fold)}
        #for j in range(n_jets): 
        #    df_bkg_dic[j]     = {}
        #    k_fold_bkg_dic[j] = {i:[] for i in range(n_fold)}
        #    k_fold_bkg[j]     = []

        df         = self.df
        df_sgn     = df[df['is_signal_new']==1]
        df_bkg_t   = df[df['is_signal_new']==0]
        #df_bkg     = {}
        #qcd_n_dic  = {}
        #for j in range(n_jets):
        #    df_bkg[j]  = df_bkg_t[df_bkg_t['jetIndex_0']==j] 
        #    qcd_n_dic[j] = dict(df_bkg[j].groupby('xs').size() ) 
        qcd_n_dic  = dict( df_bkg_t.groupby('xs').size() )
        """
        for j in range(n_jets):
            print '>>>> j'+str(j)+':'
            for key, item in qcd_n_dic[j].iteritems():
                print '#events for xs=', key, ': ', item
                #tmp_df             = df_bkg[j][df_bkg[j]['xs']==key]
                tmp_df             = (df_bkg[j][df_bkg[j]['xs']==key])[:upLmt]
                print '#events for xs=', key, '(with upper limit): ', len(tmp_df)   

                df_bkg_dic[j][key] = np.array_split(tmp_df, n_fold)
                for i in range(n_fold):    k_fold_bkg_dic[j][i].append( df_bkg_dic[j][key][i] )
        """
        for key, item in qcd_n_dic.iteritems():
            print '#events for xs=', key, ': ', item
            tmp_df             = (df_bkg_t[df_bkg_t['xs']==key])[:upLmt]
            print '#events for xs=', key, '(with upper limit): ', len(tmp_df)   

            df_bkg_dic[key]    = np.array_split(tmp_df, n_fold)
            for i in range(n_fold):    k_fold_bkg_dic[i].append( df_bkg_dic[key][i] )


 
        k_fold_sgn = np.array_split(df_sgn, n_fold)
        for i in range(n_fold):    k_fold_bkg.append( pd.concat(k_fold_bkg_dic[i]) )
        #for j in range(n_jets):
        #    for i in range(n_fold):  
        #        k_fold_bkg[j].append( pd.concat(k_fold_bkg_dic[j][i]) )

        out_dic['sgn'] = k_fold_sgn 
        out_dic['bkg'] = k_fold_bkg
        return out_dic 



    def splitTV_4(self, n_fold, upLmt=100000):
        n_jets            = 4
        df_sgn_dic        = {}
        df_bkg_dic        = {}
        k_fold_bkg_dic    = {}
        k_fold_bkg        = {}

        k_fold_sgn_dic    = {} 
        k_fold_sgn        = {}
        for j in range(n_jets):
            df_bkg_dic[j]     = {}
            k_fold_bkg_dic[j] = {i:[] for i in range(n_fold)}
            k_fold_bkg[j]     = []

            k_fold_sgn_dic[j] = {i:[] for i in range(n_fold)}
            k_fold_sgn[j]     = []

        df         = self.df
        #df_sgn     = df[df['is_signal_new']==1]
        df_sgn_t   = df[df['is_signal_new']==1]
        df_bkg_t   = df[df['is_signal_new']==0]
        df_sgn     = {} 
        df_bkg     = {}
        qcd_n_dic  = {}
        for j in range(n_jets):
            df_bkg[j]  = df_bkg_t[df_bkg_t['jetIndex_0']==j]
            qcd_n_dic[j] = dict(df_bkg[j].groupby('xs').size() )

            df_sgn[j]  = df_sgn_t[df_bkg_t['jetIndex_0']==j]


        for j in range(n_jets):
            print '>>>> j'+str(j)+':'
            for key, item in qcd_n_dic[j].iteritems():
                print '#events for xs=', key, ': ', item
                #tmp_df             = df_bkg[j][df_bkg[j]['xs']==key]
                tmp_df             = (df_bkg[j][df_bkg[j]['xs']==key])[:upLmt]
                print '#events for xs=', key, '(with upper limit): ', len(tmp_df)

                df_bkg_dic[j][key] = np.array_split(tmp_df, n_fold)  
                for i in range(n_fold):    k_fold_bkg_dic[j][i].append( df_bkg_dic[j][key][i] )
         
            df_sgn_dic[j] = np.array_split(df_sgn[j], n_fold) 
            for i in range(n_fold):    k_fold_sgn_dic[j][i].append( df_sgn_dic[j][i] ) 

        #k_fold_sgn = np.array_split(df_sgn, n_fold)

        for j in range(n_jets):
            for i in range(n_fold):
                k_fold_bkg[j].append( pd.concat(k_fold_bkg_dic[j][i]) )
                k_fold_sgn[j].append( pd.concat(k_fold_sgn_dic[j][i]) )
 
        out_dic = {}
        out_dic['sgn'] = k_fold_sgn 
        out_dic['bkg'] = k_fold_bkg
        return out_dic
        #return k_fold_sgn, k_fold_bkg[0], k_fold_bkg[1], k_fold_bkg[2], k_fold_bkg[3]





    def splitTV_sgn(self, n_fold=3, upLmt=None, n_jets=4):
        out_dic        = {}
        df             = self.df
        df_sgn         = df[df['is_signal_new']==1]
        k_fold_sgn     = np.array_split(df_sgn, n_fold)
        out_dic['sgn'] = k_fold_sgn
        return out_dic 









