'''
Package of the different pipelines used by the project.
The stage pipelines are located in a nested package pipelines.
In addition to the nested pipeline package this package contains pipeline wrapper
and pipelines helper modules.




Classes:
    src/pipeline/pipelines_helper.py:
    SupervisionFriendlyLabelBinarizer
    DataFrameSelector



Functions:

    src/pipeline/pipeline_wrapper.py:
    main_pipeline_wrapper


    src/pipeline/pipelines_helper.py module:
    Group of rebuilding dataframes methods specialized to each pipeline stage.
    the methods rebuild a df from np.ndarray (the input and output of sklearn
        pipelines).

    * rebuild_df_initial_pipeline
    * rebuild_df_attribute_adder
    * rebuild_df_feature_scaling



'''