'''
Package of different transformers used by the project pipelines.
The Transformers classes are built into the pipeline stages.

The Function 'pre_pipeline_district_removal' is called on an pre-pipeline
stage since its remove data points (rows) from the DataFrame. The function
use two internal transformers: OutliersRemover, CappedValuesRemover.


Classes:

    CombinedAttributesAdder
    FeatureScaleTransformer



Functions:
    pre_pipeline_district_removal(housing: pd.DataFrame) -> pd.DataFrame


'''