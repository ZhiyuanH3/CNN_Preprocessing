import sys
import pandas as pd
import numpy  as np
from os                import system as act
sys.path.append('./Tools/')
sys.path.append('./Classes/')
from Tools             import combi_index
from DataTypes        import pkl_df

v_str    = 'all'
pth_root = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/'
#jet_mode = '2jets'
jet_mode = 'leading_jet'
pth      = pth_root + 'raw/'       + jet_mode + '/' 
pth_out  = pth_root + 'nn_format/' + jet_mode + '/'
act('mkdir '+pth_out)

if   jet_mode == '2jets'      :    jet_str = '2j' 
elif jet_mode == 'leading_jet':    jet_str = 'j0_pfc'

xs           = { '50to100': 246300000 , '100to200': 28060000 , '200to300': 1710000 , '300to500': 351300 , '500to700': 31630 , '700to1000': 6802 , '1000to1500': 1206 , '1500to2000': 120.4 , '2000toInf': 25.25 , 'sgn': 3.782 }
qcd_cat_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
mass_list            = [20,30,40,50]
ctau_list            = [500,1000,2000,5000]
mass_sub_list        = [20,30,40]
ctau_sub_list        = [500,1000,2000]
label_str            = 'is_signal_new'
#train_val_test_ratio = {'train':0.6,'val':0.2,'test':0.2}

qcd_dict      = {}
qcd_df_dict   = {}
qcd_list      = []
sgn_dict      = {}
sgn_df_dict   = {}
sgn_list      = []
df_bkg_dict            = {}
df_sgn_dict            = {}
output_fortrain_dict   = {}
output_fortest_dict    = {}

######################################## For backgrounds:
tot_xs      = 0
for qcd_i in qcd_cat_list:
    key_i              = 'QCD_HT'+qcd_i+'_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1_'+jet_str+'_skimed'+'.h5'#'.pkl'
    inst_tmp           = pkl_df(pth,key_i) # Create instance
    qcd_dict[qcd_i]    = inst_tmp # Dictionary that stores the instances with various HT bin
    qcd_df_dict[qcd_i] = inst_tmp.df
    #inst_tmp.shift_col_num(-1) # Column starts with 'E_0'
    inst_tmp.set_xs( xs[qcd_i] ) # Set cross-sections
    tot_xs            += xs[qcd_i] # Calculate total cross-section
    inst_tmp.set_label(0, label_str) # Set the 'is_signal_new' column value
    ##################################### Collect badkgrounds from all HT bins:   
    qcd_list.append(inst_tmp.df)
    print qcd_i, '<<<<< events: ', len(inst_tmp.df)  



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
    key_i = 'VBFH_HToSSTobbbb_MH-125_MS-'+m_i+'_ctauS-'+l_i+'_TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC_'+jet_str+'_skimed'+'.h5'#'.pkl'
    inst_tmp            = pkl_df(pth,key_i)
    sgn_dict[sgn_i]     = inst_tmp
    sgn_df_dict[sgn_i]  = inst_tmp.df
    #inst_tmp.shift_col_num(-1)
    inst_tmp.set_xs( xs['sgn'] ) # Set cross-sections
    inst_tmp.set_label(1, label_str)
    # Collect signal samples from sub phase space:
    sgn_list.append(inst_tmp.df)
    print sgn_i, '<<<<< events: ', len(inst_tmp.df)




######################################## Mix the background and signals:
######################################################
# This generates the datasets for training the Model #
######################################################
#"""
# Combine all backgrounds from all HT bins:
df_bkg_dict = pd.concat(qcd_list, ignore_index=True)
# Combine all signals from the given subset of phase space:
df_sgn_dict = pd.concat(sgn_list, ignore_index=True)

print '(bkg): ', str(len(df_bkg_dict))
print '(sgn): ', str(len(df_sgn_dict))   
# Mix and shuffle signal and background:
output_fortrain_dict = pd.concat([df_bkg_dict, df_sgn_dict], ignore_index=True)
output_fortrain_dict = output_fortrain_dict.iloc[np.random.permutation(len(output_fortrain_dict))]
#"""

#####################################################
# This generates the datasets for testing the Model #
#####################################################
# Generate all train/val/test samples:
for sgn_i in mass_ctau:
    output_fortest_dict[sgn_i] = pd.concat([df_bkg_dict, sgn_df_dict[sgn_i] ], ignore_index=True)
    output_fortest_dict[sgn_i] = output_fortest_dict[sgn_i].iloc[np.random.permutation(len(output_fortest_dict[sgn_i]))]










#for sgn_i in mass_ctau:    print output_fortest_dict[sgn_i]
#exit()
######################################### Outputs:
#pth_out_train = pth_out + '/' + 'train/'
pth_out_test  = pth_out + '/' + 'test/'
#act('mkdir '+pth_out_train)
act('mkdir '+pth_out_test)

# For training:
#output_fortrain_dict[tvt_i].to_hdf( pth_out_train + 'vbf_qcd-'+tvt_i+'-'+'v0_40cs'+'.h5','table',append=False)
#print str(len( output_fortrain_dict ))

# For testing:
for sgn_i in mass_ctau:
    pth_out_test_i  = pth_out_test + '/' + sgn_i + '/'
    act('mkdir '+pth_out_test_i)
    output_fortest_dict[sgn_i].to_hdf( pth_out_test_i + 'vbf_qcd_'+v_str+'_'+'.h5','table',append=False)
    print 'num of events for ' + sgn_i + ': ' + str(len( output_fortest_dict[sgn_i] ))
















