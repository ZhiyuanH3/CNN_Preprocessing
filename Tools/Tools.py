
############################
# combinations out of sets #
############################

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















