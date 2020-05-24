

def parse_initial_estimators_results():

    # load pickle of the model dict, as used in initial_model_try
    import pickle
    with open('ml_intro_SGDR.pickle', 'rb') as handle:
        ml_intro_models_results1 = pickle.load(handle)


    # each object in the model dict is of ModelResult class, consist of a
    # detailed  __str__ method
    for key in ml_intro_models_results1:
        print(ml_intro_models_results1[key])




