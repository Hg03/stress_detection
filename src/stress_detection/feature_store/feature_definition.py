from stress_detection.scripts.utils import load_config
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Int64, String, Float64
from datetime import timedelta
import ibis
import os


def create_schemas():
    configs = load_config("feature")
    data = ibis.read_parquet(os.path.join(configs.data_loading.paths.feature_store, "data/training_data.parquet"))
    data = data.drop("event_timestamp")
    schema = data.schema()
    list_of_dtypes = {ibis.expr.datatypes.core.float64: Float64, ibis.expr.datatypes.core.string: String, ibis.expr.datatypes.core.int8: Int64, ibis.expr.datatypes.core.int64: Int64}
    return [Field(name=k, dtype=list_of_dtypes[v]) for k, v in schema.items()]

training_data_source = FileSource(
    path="data/training_data.parquet", 
    event_timestamp_column="event_timestamp")
testing_data_source = FileSource(
    path="data/testing_data.parquet", 
    event_timestamp_column="event_timestamp")

employee = Entity(
    name="employee_id",
    value_type=ValueType.STRING,
    description="Employee ID"
)

schemas = create_schemas()
employee_training_fv = FeatureView(name="employee_training_features", 
                          entities=[employee], 
                          ttl=timedelta(days=1), 
                          schema=schemas,
                          source=training_data_source)

employee_testing_fv = FeatureView(name="employee_testing_features", 
                          entities=[employee], 
                          ttl=timedelta(days=1), 
                          schema=schemas,
                          source=testing_data_source)