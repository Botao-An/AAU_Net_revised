import os
import pandas as pd
import numpy as np
def get_best_model(model_path):

    # paramater setting
    col = 3
    
    # read the train log
    log = pd.read_csv(os.path.join(model_path, 'train_log.txt'),skiprows = 0,header=None)
    acc = np.array([float(x[5:]) for x in list(log[col])])
    model_num = np.argmax(acc)

    # get the model list
    key_words = ['Generator', 'Discrim']
    model_list   = os.listdir(os.path.join(model_path, 'model'))
    
    # get the model path
    return_dict = {}
    for model in model_list:
        for key in key_words:
            if key in model and model_num+1==int(model.split('_')[2]):
                return_dict.update({key:os.path.join(model_path, 'model', model)})
    # print('best model: ', return_dict)

    return return_dict
