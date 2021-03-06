#import argparse
from   time              import time as tm
import root_numpy     as rnp
import numpy          as np
import pandas         as pd
from   ROOT              import TFile
from   sklearn.externals import joblib

from   matplotlib        import pyplot as plt
import matplotlib.colors as colors

color_lst = list(colors._colors_full_map.values())
#################
# settings      #
#################
#-------------------------------------------------------------------------------------------------------------------------
sgn_on = 1
bkg_on = 1

path        = '/beegfs/desy/user/hezhiyua/backed/fromLisa/fromBrianLLP/'
Npfc        = 400#2#400#40
#path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forBDT/' 
path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_for2d/pfc_400/raw/'
versionN_b  = 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1'
versionN_s  = 'TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC'
#cut_on      = 1
#newFileName = ''
#drop_nan    = 1
leading_jet = 1#0#1
if leading_jet:    jet_idx = 0 # >>>>>>>>>>>>>> leading jet
else          :    jet_idx = 1 # >>>>>>>>>>>>>> the 2nd jet

qcd_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
m_list   = [20,30,40,50]
l_list   = [500,1000,2000,5000] 

ftr_dict = {'jetIndex': 'jetIndex',
            'pt'      : 'pt', 
            'energy'  : 'E', 
            'px'      : 'PX', 
            'py'      : 'PY',
            'pz'      : 'PZ',
            'isTrack' : 'C',
            'pdgId'   : 'PID'}
#-------------------------------------------------------------------------------------------------------------------------
#######################################
#['cHadE','nHadE','cHadEFrac','nHadEFrac','nEmE','nEmEFrac','cEmE','cEmEFrac','cmuE','cmuEFrac','muE','muEFrac','eleE','eleEFrac','eleMulti','photonE','photonEFrac','photonMulti','cHadMulti','nHadMulti','npr','cMulti','nMulti','ecalE','nSelectedTracks','VBF_DisplacedJet40_VTightID_Hadronic_match','VBF_DisplacedJet40_VVTightID_Hadronic_match','isCaloTag','met']
#######################################

def pick_ele(x):
    if len(x) == 0:    return None
    else          :    return x[0] 

def ordering(x, n_col):
    order   = []
    ord_dic = {}
    for i in xrange(n_col):
        if x[i] <= Npfc:    ord_dic[x[i]] = i

    max_rank = max(list(ord_dic))
    cc = 1
    while cc <= max_rank:
        for key, item in ord_dic.iteritems():
            if cc == key:
                order.append(item)
                cc += 1
                break
    #for key, item in ord_dic.iteritems():
    #    if cc == key:   
    #        order.append(item)
    #        cc += 1
    leng = len(order)
    if leng < Npfc:    order = order + (Npfc-leng)*[-1]
    return order

def pick_top(x, default_val):
    order   = []
    for i in xrange(Npfc):
        i_col = int( x['o'+str(i)] )
        if pd.isnull(i_col):    break
        elif i_col == -1   :    order.append(default_val)
        else               :    order.append( x[i_col] )
    return order



def pick_ch(df):
    if (df != 211) & (df != -211) & (df != 130):    return 0#-1
    elif (df == 211) | (df == -211)            :    return 1
    elif (df == 130)                           :    return 0

def pick_h(df):
    if (df != 211) & (df != -211) & (df != 130):    return 0#-1
    else                                       :    return 1



def calc_chf(row):
    tot_e = 0
    che   = 0
    for i in range(Npfc):
        tot_e += row['E_'+str(1)] 
        ch_i   = pick_ch( row['PID_'+str(1)] )
        che   += row[e_i] * ch_i
    return che / float(tot_e)



def colm_gen(in_lst, n_pfc):
    drop_len = len('PFCandidates.')  
    out_lst  = []
    for a in in_lst:
        a_str  = a[drop_len:]
        in_str = ftr_dict[a_str] 
        for i in range(n_pfc):
            out_lst.append( in_str+'_'+str(i) )
    return out_lst




def run_nn_i(inName):

    print inName
    attr_lst     = ['jetIndex','pt','energy','px','py','pz','isTrack','pdgId']
    pre_str      = 'PFCandidates'    
    cut_pre_str  = 'Jets['+str(jet_idx)+']'
    cut_attr_lst = ['pt','eta','isGenMatched']

    if 'QCD' in inName:    is_sgn = 0
    else              :    is_sgn = 1
    f1           = TFile(path + inName, 'r')
    tin          = f1.Get('ntuple/tree')
    NumE         = tin.GetEntriesFast()
    print '\nEntries: ', NumE
    s_cut        = None

    df_s    = {}
    cut_lst = []
    ftr_lst = []
    for i in cut_attr_lst:
        tmp_str        = cut_pre_str+'.'+i 
        cut_lst.append(tmp_str) 
        df_s[tmp_str] = pd.DataFrame( rnp.tree2array(tin, [tmp_str], stop=s_cut) )
    for i in attr_lst:
        tmp_str        = pre_str+'.'+i 
        ftr_lst.append(tmp_str)
        df_s[tmp_str]        = pd.DataFrame( rnp.tree2array(tin, [tmp_str], stop=s_cut) )
    f1.Close()

    df = pd.DataFrame()
    for i in cut_lst:    df[i] = df_s[i]
    for i in ftr_lst:    df[i] = df_s[i]
    for i in cut_lst:    df[i] = df[i].apply(pick_ele)
    #if drop_nan:    df = df.dropna()
    #print len(df)
    #print df[:20]
    mask_pt  = df[cut_pre_str+'.'+'pt'] > 15
    mask_eta = (df[cut_pre_str+'.'+'eta'] > -2.4) & (df[cut_pre_str+'.'+'eta'] < 2.4)
    mask_mtc = df[cut_pre_str+'.'+'isGenMatched'] == 1
    mask_bkg = mask_pt & mask_eta
    mask_sgn = mask_bkg & mask_mtc
    
    if is_sgn: msk = mask_sgn
    else     : msk = mask_bkg 
    df       = df[msk]
    print len(df)
    #print df[:20]
    df.drop(cut_lst,axis=1)

    
    df_dict       = {}
    for a in ftr_lst:
        df_dict[a] = pd.DataFrame.from_records( df[a].values.tolist() )
        msk_jet    = df_dict[pre_str+'.'+'jetIndex']     == jet_idx # >>>> 0:leading jet
        df_dict[a] = df_dict[a][msk_jet]
    df_pt_rank    = df_dict[pre_str+'.'+'pt'].rank(axis=1, ascending=False, method='first')
    pt_rank       = df_pt_rank.copy()



    #print pt_rank
    df_count =  pt_rank.count(axis='columns')
    #print df_count
    nt_list  = df_count.values.tolist()
    #print nt_list
    n_events = len(df_count)
    #print n_events


    #exit()

    """
    n_col            = pt_rank.shape[1]
    order_pt         = pt_rank.apply(lambda row: ordering(row, n_col), axis=1)
    order_df         = pd.DataFrame.from_records( order_pt.values.tolist() )
    order_df.columns = [ 'o'+str(i) for i in range(Npfc) ]
    df_ord           = {}
    pan_list         = []

    for a in ftr_lst:#[:NumOfVecEl]: # to be checked!!! 
        if   a == pre_str+'.'+'isTrack':
            def_val = -1
            df_typ  = int
        elif a == pre_str+'.'+'pdgId':
            def_val = -1
            df_typ  = int
        else             :
            def_val = 0
            df_typ  = float
        tmp_df_ord   = pd.concat( [df_dict[a].copy(), order_df.copy()], axis=1)
        df_ord_list  = tmp_df_ord.apply(lambda row: pick_top(row, def_val), axis=1)
        df_ord[a]    = pd.DataFrame.from_records( df_ord_list.values.tolist() )
        tmp_df_ord   = df_ord[a].astype(df_typ)
        pan_list.append(tmp_df_ord)
    #print df_ord['pt'][:8]
    pan         = pd.concat(pan_list, axis=1)
    colN        = colm_gen(ftr_lst, Npfc)
    pan.columns = colN

    for i in range(Npfc):
        pan['H'+'_'+str(i)] = pan['PID'+'_'+str(i)].apply(pick_h)

    #pan['CHF']   = pan.apply(calc_chf, axis=1)
    print pan[:8]
    print 'Events: ', len(pan[:])
    
    #pan.to_hdf(path_out+'/'+inName[:-5]+'_j'+str(jet_idx)+'_skimed'+'.h5', key='df', mode='w', dropna=True)
    """

    plt.hist(nt_list, bins, alpha=0.5, color=color_lst[ci], label=label_i, normed=True)









def run_bdt_i(inName):

    print inName
    pre_str  = 'Jets['+str(jet_idx)+']'
    #attr_lst = ['cHadEFrac','nHadEFrac','cHadMulti','nHadMulti','npr','pt','eta','isGenMatched']
    attr_lst = ['pt','eta','isGenMatched','npr']
    if 'QCD' in inName:    is_sgn = 0
    else              :    is_sgn = 1
    f1       = TFile(path + inName, 'r')
    tin      = f1.Get('ntuple/tree')
    NumE     = tin.GetEntriesFast()
    print '\nEntries: ', NumE
    s_cut         = None#1000

    df_dict  = {}
    for i in attr_lst:
        tmp_str    = pre_str+'.'+i 
        df_dict[i] = pd.DataFrame( rnp.tree2array(tin, [tmp_str], stop=s_cut) )

    f1.Close()

    df = pd.DataFrame()
    for i in attr_lst:    df[i] = df_dict[i]

    for i in attr_lst:
        df[i] = df[i].apply(pick_ele) 
        #print df[i][:12]
    #if drop_nan:    df = df.dropna()
    #print len(df)
    mask_pt  = df['pt'] > 15
    mask_eta = (df['eta'] > -2.4) & (df['eta'] < 2.4)
    mask_mtc = df['isGenMatched'] == 1
    mask_bkg = mask_pt & mask_eta
    mask_sgn = mask_bkg & mask_mtc
    
    if is_sgn: msk = mask_sgn
    else     : msk = mask_bkg 
    df       = df[msk]
    print len(df)

    pan = df

    np_lst  = df['npr'].values.tolist()
    plt.hist(np_lst, bins, alpha=0.5, color=color_lst[ci], label=label_i, normed=True)
    
    #print pan
    #pan.to_hdf(path_out+'/'+inName[:-5]+'_j'+str(jet_idx)+'_skimed'+'.h5', key='df', mode='w', dropna=True)



ci   = 0 
bins = np.linspace(0,130,40) 
plt.title('Number of jet constituents')
plt.xlabel('number of PFCs per jet')
plt.ylabel('a.u.')

#"""
if sgn_on:
    for m_i in m_list:
        for l_i in l_list:

            label_i = 'VBF_'+ str(m_i) + 'GeV_' + str(l_i) + 'mm'
            in_name = 'VBFH_HToSSTobbbb_MH-125_MS-' + str(m_i) + '_ctauS-' + str(l_i) + '_' + versionN_s + '.root'
            #run_bdt_i(in_name)
            run_nn_i(in_name)    
            ci += 1
"""
if bkg_on:
    for qcd_i in qcd_list:
        
        label_i = 'QCD_HT'+qcd_i   
        in_name = 'QCD_HT'+qcd_i+'_'+versionN_b+'.root'
        #run_bdt_i(in_name)
        run_nn_i(in_name)
        ci += 1
"""

plt.legend()
plt.show()















