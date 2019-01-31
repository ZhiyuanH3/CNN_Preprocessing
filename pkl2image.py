import sys
import pandas as pd
import numpy  as np
from os                import system as act
sys.path.append('./Tools/')
sys.path.append('./Classes/')
from Tools             import combi_index
from DataTypes        import pkl_df

#version = 0 
pth     = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/pfc_400/large_sgn/'
pth_out = pth + '/' + 'output/'
act('mkdir '+pth_out)

xs           = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }
qcd_cat_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
#qcd_cat_list = ['100to200','200to300']
mass_list            = [20,30,40,50]
ctau_list            = [500,1000,2000,5000]
mass_sub_list        = [20,30,40]
ctau_sub_list        = [500,1000,2000]
label_str            = 'is_signal_new'
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
df_bkg_dict            = {}
df_sgn_dict            = {}
output_fortrain_dict   = {}
output_fortest_dict    = {}

######################################## For backgrounds:
tot_xs      = 0
for qcd_i in qcd_cat_list:
    key_i              = 'QCD_HT'+qcd_i+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1_1j_skimed.pkl'
    inst_tmp           = pkl_df(pth,key_i) # Create instance
    qcd_dict[qcd_i]    = inst_tmp # Dictionary that stores the instances with various HT bin
    qcd_df_dict[qcd_i] = {}

    inst_tmp.shift_col_num(-1) # Column starts with 'E_0'
 
    inst_tmp.set_xs( xs[qcd_i] ) # Set cross-sections
    tot_xs            += xs[qcd_i] # Calculate total cross-section
    inst_tmp.set_label(0, label_str) # Set the 'is_signal_new' column value
    tmp_df_dict        = inst_tmp.split(train_val_test_ratio) # Split the dataframe into train/val/test sections
    for tvt_i in train_val_test_ratio:
        qcd_df_dict[qcd_i][tvt_i] = tmp_df_dict[tvt_i]
        qcd_df_dict[qcd_i][tvt_i] = qcd_df_dict[qcd_i][tvt_i].assign(weight=0) # Create 'weight' column

############################ Set weights:
for qcd_i in qcd_cat_list:
    tmpt         = qcd_dict[qcd_i]
    tmp_df       = qcd_df_dict[qcd_i]
    for tvt_i in train_val_test_ratio:
        weight_tvt_i                      = tmpt.xs / float( tot_xs * tmpt.read_n(tvt_i) ) # Calculate weights
        (tmp_df[tvt_i]).loc[:,['weight']] = weight_tvt_i # Set 'weight' column
        print weight_tvt_i  
        ##################################### Collect badkgrounds from all HT bins:   
        print '<<<<<', len(tmp_df[tvt_i])  
        qcd_list_dict[tvt_i].append(tmp_df[tvt_i])




######################################## For signals(similar to background):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var_dic         = {}
var_dic['mass'] = mass_list
var_dic['ctau'] = ctau_list
var_cmb_lst     = combi_index( (len(mass_list),len(ctau_list)) )
# To generate all combinations:
combi           = [  [ var_dic[k[1]][i[k[0]]] for k in enumerate(var_dic) ] for i in var_cmb_lst  ]
mass_ctau       = []
m_c_tpl         = []
for i in combi: 
    m_i     = str(i[1])
    l_i     = str(i[0])
    sgn_i   = m_i+'_'+l_i
    mass_ctau.append(sgn_i)
    m_c_tpl.append( (m_i,l_i) )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
sub_dic         = {}
sub_dic['mass'] = mass_sub_list
sub_dic['ctau'] = ctau_sub_list
sub_cmb_lst     = combi_index( (len(mass_sub_list),len(ctau_sub_list)) )
# To generate all combinations:
sub_combi       = [  [ sub_dic[k[1]][i[k[0]]] for k in enumerate(sub_dic) ] for i in sub_cmb_lst  ]
sub_mass_ctau   = []
sub_m_c_tpl     = []
for i in sub_combi:
    m_i     = str(i[1])
    l_i     = str(i[0])
    sgn_i   = m_i+'_'+l_i
    sub_mass_ctau.append(sgn_i)
    sub_m_c_tpl.append( (m_i,l_i) )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







for tpl_i in m_c_tpl:
    m_i = tpl_i[0]
    l_i = tpl_i[1] 
    sgn_i = m_i + '_' + l_i
    key_i = 'VBFH_HToSSTobbbb_MH-125_MS-'+m_i+'_ctauS-'+l_i+'_TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC_1j_skimed.pkl'
    inst_tmp            = pkl_df(pth,key_i)
    sgn_dict[sgn_i]     = inst_tmp
    sgn_df_dict[sgn_i]  = {}
     
    inst_tmp.shift_col_num(-1)

    inst_tmp.set_label(1, label_str)
    tmp_df_dict         = inst_tmp.split(train_val_test_ratio)
    for tvt_i in train_val_test_ratio:
        sgn_df_dict[sgn_i][tvt_i] = tmp_df_dict[tvt_i]
        sgn_df_dict[sgn_i][tvt_i] = sgn_df_dict[sgn_i][tvt_i].assign(weight=0)

        weight_tvt_i                                  = 1. / float( tmpt.read_n(tvt_i) )
        (sgn_df_dict[sgn_i][tvt_i]).loc[:,['weight']] = weight_tvt_i
        print weight_tvt_i
 

# Collect signal samples from sub phase space:
for sgn_i in sub_mass_ctau:
    for tvt_i in train_val_test_ratio:
        sgn_list_dict[tvt_i].append(sgn_df_dict[sgn_i][tvt_i])






######################################## Mix the background and signals:
######################################################
# This generates the datasets for training the Model #
######################################################
#"""
for tvt_i in train_val_test_ratio:
    # Combine all backgrounds from all HT bins:
    df_bkg_dict[tvt_i] = pd.concat(qcd_list_dict[tvt_i], ignore_index=True)
    # Combine all signals from the given subset of phase space:
    df_sgn_dict[tvt_i] = pd.concat(sgn_list_dict[tvt_i], ignore_index=True)
    print tvt_i+'(bkg): ', str(len(df_bkg_dict[tvt_i]))
    print tvt_i+'(sgn): ', str(len(df_sgn_dict[tvt_i]))   
    # Mix and shuffle signal and background:
    output_fortrain_dict[tvt_i] = pd.concat([df_bkg_dict[tvt_i], df_sgn_dict[tvt_i]], ignore_index=True)
    output_fortrain_dict[tvt_i] = output_fortrain_dict[tvt_i].iloc[np.random.permutation(len(output_fortrain_dict[tvt_i]))]
#"""

#####################################################
# This generates the datasets for testing the Model #
#####################################################
#"""
# Generate all train/val/test samples:
for sgn_i in mass_ctau:
    output_fortest_dict[sgn_i]            = {}
    for tvt_i in train_val_test_ratio:
        output_fortest_dict[sgn_i][tvt_i] = pd.concat([df_bkg_dict[tvt_i], sgn_df_dict[sgn_i][tvt_i] ], ignore_index=True)
        output_fortest_dict[sgn_i][tvt_i] = output_fortest_dict[sgn_i][tvt_i].iloc[np.random.permutation(len(output_fortest_dict[sgn_i][tvt_i]))]
"""
# Generate only test samples:
for sgn_i in mass_ctau:
    output_fortest_dict[sgn_i] = pd.concat([df_bkg_dict['test'], sgn_df_dict[sgn_i]['test'] ], ignore_index=True)
    output_fortest_dict[sgn_i] = output_fortest_dict[sgn_i].iloc[np.random.permutation(len(output_fortest_dict[sgn_i]))]
"""


######################################### Debug:
for i in qcd_cat_list:
    ttt = qcd_df_dict[i]
    tt = ttt['val'] 
    t  = ttt['test']
    print tt[:2]
    print t[:2]
    #print list(t.columns.values)
for i in sub_mass_ctau:
    ttt = sgn_df_dict[i]
    tt = ttt['val']
    t  = ttt['test']
    print tt[:2]
    print t[:2]



#exit()
######################################### Outputs:
# For training:
pth_out_train = pth_out + '/' + 'train/'
pth_out_test  = pth_out + '/' + 'test/'
act('mkdir '+pth_out_train)
act('mkdir '+pth_out_test)
for tvt_i in train_val_test_ratio:
    #output_fortrain_dict[tvt_i].to_hdf( pth_out_train + 'vbf_qcd-'+tvt_i+'-'+'v0_40cs'+'.h5','table',append=True)
    print tvt_i + ': ' + str(len( output_fortrain_dict[tvt_i] ))


# For testing:
for sgn_i in mass_ctau:
    pth_out_test_i  = pth_out_test + '/' + sgn_i + '/'
    act('mkdir '+pth_out_test_i)
    for tvt_i in train_val_test_ratio:
        #output_fortest_dict[sgn_i][tvt_i].to_hdf( pth_out_test_i + 'vbf_qcd-'+tvt_i+'-'+'v0_40cs'+'.h5','table',append=True)
        print sgn_i + '/' + tvt_i + ': ' + str(len( output_fortest_dict[sgn_i][tvt_i] ))
















