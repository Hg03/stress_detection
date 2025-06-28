from stress_detection.scripts.utils import load_config
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Int64, String, Float64
from datetime import timedelta
import ibis
import os

configs = load_config("feature")
def create_schemas(train_or_test: str = "train"):
    if train_or_test == "train":
        data = ibis.read_parquet(os.path.join(configs.data_loading.paths.feature_store, configs.data_loading.paths.training_data))
    elif train_or_test == "test":
        data = ibis.read_parquet(os.path.join(configs.data_loading.paths.feature_store, configs.data_loading.paths.testing_data))
    data = data.drop(configs.data_loading.columns.event_timestamp)
    schema = data.schema()
    list_of_dtypes = {ibis.expr.datatypes.core.float64: Float64, ibis.expr.datatypes.core.string: String, ibis.expr.datatypes.core.int8: Int64, ibis.expr.datatypes.core.int64: Int64}
    return [Field(name=k, dtype=list_of_dtypes[v]) for k, v in schema.items()]

training_data_source = FileSource(
    path=configs.data_loading.paths.training_data,
    event_timestamp_column=configs.data_loading.columns.event_timestamp)
testing_data_source = FileSource(
    path=configs.data_loading.paths.testing_data,
    event_timestamp_column=configs.data_loading.columns.event_timestamp)

entity = Entity(
    name=configs.data_loading.columns.unique_key,
    value_type=ValueType.STRING,
    description="Employee ID"
)

schemas = create_schemas()
training_fv = FeatureView(name="training_features", 
                          entities=[entity],
                          ttl=timedelta(days=1), 
                          schema=schemas,
                          source=training_data_source)

testing_fv = FeatureView(name="testing_features", 
                          entities=[entity],
                          ttl=timedelta(days=1), 
                          schema=schemas,
                          source=testing_data_source)