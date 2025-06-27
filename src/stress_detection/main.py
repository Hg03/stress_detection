import ibis

if __name__ == "__main__":
    sample_data = ibis.table({"id": "int64", "value": "float64"}, name="sample_data")
    print(sample_data.head())