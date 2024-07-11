import csv

# Specify the input and output file paths
input_file = 'corpus/TopiOCQA/full_wiki_segments.tsv'
output_file = 'corpus/TopiOCQA/full_wiki_segments_n1000.tsv'

# Number of rows to extract (excluding header)
num_rows_to_extract = 1000  # Adjust this number as needed

# Open the input file in read mode and output file in write mode
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    # Create a CSV reader and writer
    tsv_reader = csv.reader(infile, delimiter='\t')
    tsv_writer = csv.writer(outfile, delimiter='\t')
    
    # Read the header from the input file and write it to the output file
    header = next(tsv_reader)
    tsv_writer.writerow(header)
    
    # Write a specified number of rows to the output file
    for i, row in enumerate(tsv_reader):
        if i < num_rows_to_extract:
            tsv_writer.writerow(row)
        else:
            break

print(f'Extracted {num_rows_to_extract} rows (excluding header) from {input_file} and saved to {output_file}.')
