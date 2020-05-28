

def parse_initial_estimators_results():

    # load pickle of the model dict, as used in initial_model_try
    import pickle
    with open('src\grid_search_extra_trees_v3_no_capp.pickle', 'rb') as handle:
        extra_trees = pickle.load(handle)

    print(extra_trees.best_estimator_)
    # each object in the model dict is of ModelResult class, consist of a
    # detailed  __str__ method
    for key in models_dict:
        print(models_dict[key])




