import pandas as pd
import numpy  as np
from os                import system as act
from sklearn.externals import joblib


class pkl_df:   
    def __init__(self,pth,fn):
        self.name    = fn
        self.path    = pth
        self.df      = joblib.load( self.path + '/' +self.name )  
        firstColName = (self.df).columns.values[0]
        dropone      = (self.df)[firstColName] != -1
        self.df      = (self.df)[dropone]
        self.events  = len(self.df)
        self.train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}

    def set_xs(self,xs):
        self.xs       = xs
        self.df['xs'] = xs

    def set_label(self,label):
        self.label           = label
        #self.df['is_signal'] = label   
        self.df['is_signal_new'] = label

    def split(self,train_val_test_ratio):
        self.train_val_test_ratio = train_val_test_ratio
        self.n_train              = int( self.events * train_val_test_ratio['train'] )
        self.n_val                = int( self.events * train_val_test_ratio['val']   )
        self.n_test               = self.events - self.n_train - self.n_val 
        self.df_train             = (self.df).iloc[:self.n_train] 
        self.df_val               = (self.df).iloc[self.n_train:self.n_train+self.n_val]
        self.df_test              = (self.df).iloc[self.n_train+self.n_val:]        
        df_dict = {'train': self.df_train, 'val': self.df_val, 'test': self.df_test}
        return df_dict#self.df_train, self.df_val, self.df_test

    def read_n(self,tvt_str):
        if   tvt_str == 'train': return self.n_train
        elif tvt_str == 'val'  : return self.n_val
        elif tvt_str == 'test' : return self.n_test
    
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

    def set_weight(self,weight):
        self.df_train['weight'] = weight['train']
        self.df_val['weight']   = weight['val']
        self.df_test['weight']  = weight['test']
        

def combi_index(tpl):
    tpl_lng  = len(tpl)
    end_n    = 1
    for i in range(tpl_lng):    end_n *= tpl[i]
    combi    = [[0 for i in range(tpl_lng)]]
    indx_dic = {}
    for i in range(tpl_lng):    indx_dic[i] = 0
    for j in range(end_n-1):
        for i in range(tpl_lng):
            if   i==0:
                if indx_dic[i] != tpl[i]-1:
                    chng_bool    = 0
                    indx_dic[i] += 1
                else:
                    chng_bool    = 1
                    indx_dic[i]  = 0
            else:
                if indx_dic[i] != tpl[i]-1:
                    if indx_dic[i-1]==0 and chng_bool:
                        chng_bool    = 1
                        indx_dic[i] += 1
                    else:
                        chng_bool    = 0
                else:
                    if indx_dic[i-1]==0 and chng_bool:
                        chng_bool    = 1
                        indx_dic[i]  = 0
                    else:
                        chng_bool    = 0
        tmp_cmb = [indx_dic[i] for i in range(tpl_lng)]
        combi.append(tmp_cmb)
    return combi


version = 0 
pth     = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/pfc_400/large_sgn/'
pth_out = pth + '/' + 'output/'
act('mkdir '+pth_out)

xs           = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }
#qcd_cat_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
qcd_cat_list = ['100to200','200to300']
mass_list    = [20,30,40]#[20,30,40,50]
ctau_list    = [500,1000,2000]#[500,1000,2000,5000]

train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}

qcd_dict      = {}
qcd_df_dict   = {}
qcd_list_dict = {}
sgn_dict      = {}
sgn_df_dict   = {}
sgn_list_dict = {}
for tvt_i in train_val_test_ratio:
    sgn_list_dict[tvt_i] = []
    qcd_list_dict[tvt_i] = []
df_bkg_dict   = {}
df_sgn_dict   = {}
output_dict   = {}



tot_xs      = 0
for qcd_i in qcd_cat_list:
    key_i              = 'QCD_HT'+qcd_i+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1_1j_skimed.pkl'
    inst_tmp           = pkl_df(pth,key_i)
    qcd_dict[qcd_i]    = inst_tmp
    qcd_df_dict[qcd_i] = {}

    inst_tmp.shift_col_num(-1)
 
    inst_tmp.set_xs( xs[qcd_i] )
    tot_xs            += xs[qcd_i] 
    inst_tmp.set_label( 0 )
    tmp_df_dict        = inst_tmp.split(train_val_test_ratio) 
    for tvt_i in train_val_test_ratio:
        qcd_df_dict[qcd_i][tvt_i] = tmp_df_dict[tvt_i]
        qcd_df_dict[qcd_i][tvt_i] = qcd_df_dict[qcd_i][tvt_i].assign(weight=0) 


for qcd_i in qcd_cat_list:
    tmpt         = qcd_dict[qcd_i]
    tmp_df       = qcd_df_dict[qcd_i]
    for tvt_i in train_val_test_ratio:
        weight_tvt_i                      = tmpt.xs / float( tot_xs * tmpt.read_n(tvt_i) )
        (tmp_df[tvt_i]).loc[:,['weight']] = weight_tvt_i
        print weight_tvt_i  

        qcd_list_dict[tvt_i].append(tmp_df[tvt_i])





dic         = {}
dic['mass'] = mass_list
dic['ctau'] = ctau_list
cmb_lst     = combi_index( (len(mass_list),len(ctau_list)) )
combi       = [  [ dic[k[1]][i[k[0]]] for k in enumerate(dic) ] for i in cmb_lst  ]

mass_ctau   = []
for i in combi:
    m_i   = str(i[1])
    l_i   = str(i[0])
    sgn_i = m_i+'_'+l_i
    mass_ctau.append(sgn_i)
    key_i = 'VBFH_HToSSTobbbb_MH-125_MS-'+m_i+'_ctauS-'+l_i+'_TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC_1j_skimed.pkl'
    inst_tmp            = pkl_df(pth,key_i)
    sgn_dict[sgn_i]     = inst_tmp
    sgn_df_dict[sgn_i]  = {}
     
    inst_tmp.shift_col_num(-1)
    inst_tmp.set_label( 1 )
    tmp_df_dict         = inst_tmp.split(train_val_test_ratio)
    for tvt_i in train_val_test_ratio:
        sgn_df_dict[sgn_i][tvt_i] = tmp_df_dict[tvt_i]
        sgn_df_dict[sgn_i][tvt_i] = sgn_df_dict[sgn_i][tvt_i].assign(weight=0)

        weight_tvt_i                                  = 1. / float( tmpt.read_n(tvt_i) )
        (sgn_df_dict[sgn_i][tvt_i]).loc[:,['weight']] = weight_tvt_i
        print weight_tvt_i
 
        sgn_list_dict[tvt_i].append(sgn_df_dict[sgn_i][tvt_i])




for tvt_i in train_val_test_ratio:
    df_bkg_dict[tvt_i] = pd.concat(qcd_list_dict[tvt_i], ignore_index=True)
    df_sgn_dict[tvt_i] = pd.concat(sgn_list_dict[tvt_i], ignore_index=True)
   
    output_dict[tvt_i] = pd.concat([df_bkg_dict[tvt_i], df_sgn_dict[tvt_i]], ignore_index=True)
    output_dict[tvt_i] = output_dict[tvt_i].iloc[np.random.permutation(len(output_dict[tvt_i]))]







for i in qcd_cat_list:
    ttt = qcd_df_dict[i]
    tt = ttt['val'] 
    t  = ttt['test']
    print tt[:4]
    print t[:2]
    #print list(t.columns.values)

for i in mass_ctau:
    ttt = sgn_df_dict[i]
    tt = ttt['val']
    t  = ttt['test']
    print tt[:4]
    print t[:2]



for tvt_i in train_val_test_ratio:
    output_dict[tvt_i].to_hdf( pth_out + 'vbf_qcd-'+tvt_i+'-{0}.h5'.format(version),'table',append=True)

"""
train.to_hdf( pth_out + 'vbf_qcd-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf(  pth_out + 'vbf_qcd-test-{0}.h5'.format(version) ,'table',append=True)
val.to_hdf(   pth_out + 'vbf_qcd-val-{0}.h5'.format(version)  ,'table',append=True)
"""






