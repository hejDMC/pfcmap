
cdict = {'rA':'grey','rB':'k',\
         'c1':'firebrick','c2':'dodgerblue','c3':'goldenrod','c4':'darkviolet'}

c_empty = 'w'


Adom_c23Imbalance = {'reg_prob_dict':{'rA':0.7},\
                      'cat_labels':['c1','c2','c3'],\
                      'cond_dict':{('c1','rA'):0.6, \
                                 ('c1','rB'):0.6, \
                                 ('c2','rA'):0.2, \
                                 ('c2','rB'):0.3}}#(c1|rA)}


Bdom_c23Imbalance = {'reg_prob_dict':{'rA':0.3},\
                      'cat_labels':['c1','c2','c3'],\
                      'cond_dict':{('c1','rA'):0.2, \
                                 ('c1','rB'):0.2, \
                                 ('c2','rA'):0.6, \
                                 ('c2','rB'):0.7}}#(c1|rA)}

Adom_c1dom = {'reg_prob_dict':{'rA':0.7},\
                      'cat_labels':['c1','c2'],\
                      'cond_dict':{('c1','rA'):0.6, \
                                 ('c1','rB'):0.6}}#(c1|rA)}


Bdom_c2dom = {'reg_prob_dict':{'rA':0.3},\
                      'cat_labels':['c1','c2'],\
                      'cond_dict':{('c1','rA'):0.2, \
                                 ('c1','rB'):0.2}}#(c1|rA)}