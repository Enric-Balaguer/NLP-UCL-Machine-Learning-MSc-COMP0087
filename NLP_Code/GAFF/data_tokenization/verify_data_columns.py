import pyarrow.ipc as ipc

# Replace this with the path to one of your Arrow files
arrow_file = '/home/ucabcfj/.cache/huggingface/datasets/open-phi___textbooks/default/0.0.0/292aaae99cbecacad50f692d7327887f05dacaf2/textbooks-train.arrow'

# Open the Arrow file and read the schema
with open(arrow_file, 'rb') as f:
    reader = ipc.open_stream(f)
    schema = reader.schema

# Print the column names
print(schema.names)
