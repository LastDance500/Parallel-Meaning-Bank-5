def concatenate_files(file1, file2, output_file):
    """Concatenate the contents of two files into a third file."""
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2, open(output_file, 'w', encoding='utf-8') as out:
        # Write the contents of the first file to the output file
        out.write(f1.read())
        # Write the contents of the second file to the output file
        out.write(f2.read())

lang = "nl"
# Example usage
file1 = f'/Users/xiaozhang/code/PMB5.0.0/data/pmb-5.1.0/seq2seq/{lang}/train/gold_silver.sbn'
file2 = f'/Users/xiaozhang/code/PMB5.0.0/data/pmb-5.1.0/seq2seq/{lang}/train/copper.sbn'
output_file = f'/Users/xiaozhang/code/PMB5.0.0/data/pmb-5.1.0/seq2seq/{lang}/train/gold_silver_copper.sbn'

concatenate_files(file1, file2, output_file)

print(f"Files {file1} and {file2} have been concatenated into {output_file}")
