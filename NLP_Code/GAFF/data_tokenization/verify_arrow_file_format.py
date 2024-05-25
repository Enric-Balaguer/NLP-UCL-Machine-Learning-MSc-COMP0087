import pyarrow.parquet as pq
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow as pa

# Replace with your actual file name
file_name = '/home/ucabcfj/.cache/huggingface/datasets/open-phi___textbooks/default/0.0.0/292aaae99cbecacad50f692d7327887f05dacaf2/textbooks-train.arrow'

# Attempt to read as Parquet
try:
    pq.read_table(file_name)
    print(f"The file '{file_name}' is in Parquet format.")
except Exception as e:
    print(f"Failed to read '{file_name}' as Parquet: {e}")

# Attempt to read as Feather
try:
    feather.read_table(file_name)
    print(f"The file '{file_name}' is in Feather format.")
except Exception as e:
    print(f"Failed to read '{file_name}' as Feather: {e}")

# Attempt to read as IPC (Arrow Streaming Format)
try:
    with open(file_name, 'rb') as f:
        reader = ipc.open_stream(f)
        _ = reader.read_all()
    print(f"The file '{file_name}' is in Arrow IPC Streaming format.")
except Exception as e:
    print(f"Failed to read '{file_name}' as Arrow IPC Streaming format: {e}")
