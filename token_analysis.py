from transformers import RobertaTokenizer
import csv
import matplotlib.pyplot as plt

# Load the Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# File paths
file_path = '/Users/manansuri/Desktop/Research/btp/sample/dailydialog_sample.txt'  # Replace with the path to your text file
csv_file_path = file_path.replace('.txt', '.csv')
plot_file_path = file_path.replace('.txt', '_distribution.png')

# Open and read the text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Tokenize each line, count the tokens, and create CSV
token_counts = []
lines_over_500 = 0

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Line', 'Token Count', 'Text Content'])
    for idx, line in enumerate(lines):
        tokens = tokenizer(line)['input_ids']
        token_count = len(tokens)
        token_counts.append(token_count)
        if token_count > 500:
            lines_over_500 += 1
        csv_writer.writerow([idx + 1, token_count, line.strip()])

# Plot distribution of tokens and save the plot
plt.figure(figsize=(8, 6))
plt.hist(token_counts, bins=50, color='skyblue')
plt.axvline(x=500, color='red', linestyle='--', label='500 tokens')
plt.title('Distribution of Tokens per Line')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(plot_file_path)

# Calculate the number of lines with tokens > 500 and percentage
percentage_over_500 = (lines_over_500 / len(lines)) * 100
print(f"Number of lines with tokens > 500: {lines_over_500}")
print(f"Percentage of total lines: {percentage_over_500:.2f}%")
