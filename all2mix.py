import sys
import pandas as pd
import numpy  as np
from os                import system as act
sys.path.append('./Tools/')
sys.path.append('./Classes/')
from Tools             import combi_index
from DataTypes         import pkl_df

k_fold   = 10
#v_str    = 'all_400pfc'
in_name  = 'vbf_qcd_all_400pfc' #'qcd_all_400pfc'
pth_root = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/'
#jet_mode = '2jets'
jet_mode = 'leading_jet'
#pth      = pth_root + 'nn_format/' + jet_mode + '/' 
pth      = pth_root + 'nn_format/mix/' + jet_mode + '/test/50_5000/'
pth_out  = pth + '/' + 'mixing/'#pth_root + 'nn_format/' + jet_mode + '/'
act('mkdir '+pth_out)

if   jet_mode == '2jets'      :    jet_str = '2j' 
elif jet_mode == 'leading_jet':    jet_str = 'j0_pfc'

#xs           = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }
#qcd_cat_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
mass_list            = [20,30,40,50]
ctau_list            = [500,1000,2000,5000]
mass_sub_list        = [20,30,40]
ctau_sub_list        = [500,1000,2000]
#label_str            = 'is_signal_new'
#train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}


df_out = []


df_qcd_inst = pkl_df(pth, 'vbf_qcd_all_400pfc.h5')
#df_qcd_inst = pkl_df(pth, 'qcd_all_400pfc.h5')
#df_qcd      = df_qcd_inst.df
#print df_qcd
df_sgn_dic, df_qcd_dic = df_qcd_inst.splitN(k_fold)
for i in range(k_fold):
    print df_sgn_dic[i].shape
    print df_qcd_dic[i].shape 

    tmp_df_mix = pd.concat( [df_sgn_dic[i], df_qcd_dic[i]] ) 
    df_out.append( tmp_df_mix )
    tmp_df_mix.to_hdf(pth_out+in_name+'k'+str(i)+'.h5', key='df', mode='w', dropna=False)

    






























