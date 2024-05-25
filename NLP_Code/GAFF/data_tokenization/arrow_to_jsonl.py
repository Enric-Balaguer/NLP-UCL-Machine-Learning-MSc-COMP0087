import json
import pyarrow.ipc as ipc
import pyarrow as pa

# List of your Arrow files
file1 = "/home/ucabcfj/.cache/huggingface/datasets/open-phi___textbooks/default/0.0.0/292aaae99cbecacad50f692d7327887f05dacaf2/textbooks-train.arrow"

arrow_files = [file1]

# Output JSONL file
output_jsonl_file = '/home/ucabcfj/never_lose_hope/datasets/textbooks.jsonl'

# Process each Arrow file
with open(output_jsonl_file, 'w') as jsonl_out:
    for arrow_file in arrow_files:
        with open(arrow_file, 'rb') as f:
            reader = ipc.open_stream(f)
            # Read the batches
            for batch in reader:
                # Convert to Pandas DataFrame for easier manipulation
                df = batch.to_pandas()
                # Assuming the text content is in a column named 'text', adjust if necessary
                for index, row in df.iterrows():
                    # Create a dictionary with the text key
                    doc = {'text': row['markdown']}
                    # Write the document to the JSONL file
                    jsonl_out.write(json.dumps(doc) + '\n')

print("Conversion complete.")
