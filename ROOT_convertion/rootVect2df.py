#import argparse
from   time              import time as tm
import root_numpy     as rnp
import numpy          as np
import pandas         as pd
from   ROOT              import TFile
from   sklearn.externals import joblib
#################
# settings      #
#################
path                  = '/beegfs/desy/user/hezhiyua/backed/fromLisa/fromBrianLLP/'
num_of_jets           = 1
testOn                = 0
numOfEntriesToScan    = 100 #only when testOn = 1
Npfc                  = 400#44#88#400#40
path_out    = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/Skim/fromBrian_forBDT/' 
versionN_b  = 'TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1'
versionN_s  = 'TuneCUETP8M1_13TeV-powheg-pythia8_Tranche2_PRIVATE-MC'
lola_on     = 0 # 1: prepared for lola
cut_on      = 1
newFileName = ''

#-------------------------------------
#-------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#######################################
#######################################
#['cHadE','nHadE','cHadEFrac','nHadEFrac','nEmE','nEmEFrac','cEmE','cEmEFrac','cmuE','cmuEFrac','muE','muEFrac','eleE','eleEFrac','eleMulti','photonE','photonEFrac','photonMulti','cHadMulti','nHadMulti','npr','cMulti','nMulti','ecalE','nSelectedTracks','VBF_DisplacedJet40_VTightID_Hadronic_match','VBF_DisplacedJet40_VVTightID_Hadronic_match','isCaloTag','met']
#######################################

drop_nan = 1

leading_jet = 1
if leading_jet:    jet_idx = 0
else          :    jet_idx = 1 # ???


attr_list0   = ['jetInd','energy','px','py','pz','c','pdgId','ppt']
#attr_lst     = ['jetIndex','energy','px','py','pz','isTrack','pdgId']

attr_lst = ['cHadEFrac','nHadEFrac','cHadMulti','nHadMulti','npr','pt','eta','isGenMatched']

#pre_str  = 'PFCandidates'
pre_str = 'Jets['+str(jet_idx)+']'
#pre_str = 'Jets[0]'

def run_i(inName):
    print inName
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

    df        = pd.DataFrame()
    for i in attr_lst:
        df[i] = df_dict[i]

    """
    df            = df[ df['energy'].apply(len) != 0 ]
    df['ppt']     = df['px'].pow(2) + df['py'].pow(2) # to be optimized!!!!
    df_dict       = {}
    for a in attr_list0:
        df_dict[a] = pd.DataFrame.from_records( df[a].values.tolist() )
        jet_one    = df_dict['jetInd']     == 0 #1? >>>> 0:leading jet
        df_dict[a] = df_dict[a][jet_one]
    df_dict['pt'] = df_dict['ppt'].pow(1./2)
    df_pt_rank    = df_dict['pt'].rank(axis=1, ascending=False)
    pt_rank       = df_pt_rank.copy()
    """

    def pick_ele(x):
        if len(x) == 0:    return None
        else          :    return x[0] 

    for i in attr_lst:
        df[i] = df[i].apply(pick_ele) 
        #print df[i][:12]

    if drop_nan:    df = df.dropna()
    print len(df)

    mask_pt  = df['pt'] > 15
    mask_eta = (df['eta'] > -2.4) & (df['eta'] < 2.4)
    mask_mtc = df['isGenMatched'] == 1
    mask_bkg = mask_pt & mask_eta
    mask_sgn = mask_bkg & mask_mtc
    
    if is_sgn: msk = mask_sgn
    else     : msk = mask_bkg 
    df       = df[msk]

    print len(df)


    """
    def ordering(x, n_col):
        order   = []
        ord_dic = {}
        for i in xrange(n_col):
            if x[i] <= Npfc:    ord_dic[x[i]] = i
        cc = 1
        for key, item in ord_dic.iteritems():
            if cc == key:    order.append(item)
            cc += 1
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

    n_col            = pt_rank.shape[1]
    order_pt         = pt_rank.apply(lambda row: ordering(row, n_col), axis=1)
    order_df         = pd.DataFrame.from_records( order_pt.values.tolist() )
    order_df.columns = [ 'o'+str(i) for i in range(Npfc) ]
    df_ord           = {}
    pan_list         = []
    attr_list        = ['energy','px','py','pz','c','pdgId','pt'] # pt not needed here

    for a in attr_list[:NumOfVecEl]: # to be checked!!!
        if   a == 'c'    :
            def_val = -1
            df_typ  = int
        elif a == 'pdgId':
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
    #forLola.panColNameListGen()
    #colN        = forLola.panColNameList
    pan.columns = colN
    print pan[:8]
    print 'Events: ', len(pan[:])
    
    
    def pick_ch(df):
        if (df != 211) & (df != -211) & (df != 130):
            return 0#-1
        elif (df == 211) | (df == -211):
            return 1
        elif (df == 130): 
            return 0

    def calc_e(row):
        tot_e = 0
        for i in range(Npfc):
            coln_i   = 'E_'+str(i+1)
            tot_e   += row[coln_i] 
        return tot_e

    def calc_ce(row):
        ce = 0
        for i in range(Npfc):
            e_i   = 'E_'+str(i+1)   
            c_i   = 'C_'+str(i+1)
            ce   += row[e_i] * row[c_i]   
        return ce

    def calc_che(row):
        che = 0
        for i in range(Npfc):
            e_i    = 'E_'+str(i+1)
            pid_i  = 'PID_'+str(i+1)
            ch_i   = row[pid_i]
            ch_i   = pick_ch(ch_i)
            che   += row[e_i] * ch_i
        return che


    def calc_chf(row):
        return row['che'] / float(row['tot_e'])#row['ce'] / float(row['tot_e'])

    pan['tot_e'] = pan.apply(calc_e, axis=1)
    pan['che']   = pan.apply(calc_che, axis=1)
    #pan['ce']    = pan.apply(calc_ce, axis=1)
    pan['CHF']   = pan.apply(calc_chf, axis=1)
    print pan['CHF']
    """


    #exit() 
    #pan.to_hdf(path_out+'/'+inName[:-5]+'_1j_skimed'+'.h5', key='df', mode='w', dropna=True)





qcd_list = ['100to200','200to300','300to500','500to700','700to1000','1000to1500','1500to2000','2000toInf']
m_list   = [20,30,40,50]
l_list   = [500,1000,2000,5000] 

#"""
for m_i in m_list:
    for l_i in l_list:
        in_name = 'VBFH_HToSSTobbbb_MH-125_MS-' + str(m_i) + '_ctauS-' + str(l_i) + '_' + versionN_s + '.root'
        run_i(in_name)
#"""
"""
for qcd_i in qcd_list:
    in_name = 'QCD_HT'+qcd_i+'_'+versionN_b+'.root'
    run_i(in_name)
"""




















