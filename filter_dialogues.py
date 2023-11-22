import csv

def filter_lines_by_threshold(csv_file, in_file, out_file, threshold=500):
    lines_to_keep = []
    
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip header
        for row in csv_reader:
            line_number, token_count, _ = row
            if int(token_count) <= threshold:
                lines_to_keep.append(int(line_number))

    with open(in_file, 'r') as infile, open(out_file, 'w') as outfile:
        for i, line in enumerate(infile, start=1):
            if i in lines_to_keep:
                outfile.write(line)

# Example usage:
split = "sample"
csv_file_path = f'/Users/manansuri/Desktop/Research/btp/sample/dailydialog_sample.csv'
in_file_1 = f"/Users/manansuri/Desktop/Research/btp/sample/dailydialog_sample.txt"
out_file_1 = f'/Users/manansuri/Desktop/Research/btp/sample/dailydialog_filter.txt'
# in_file_1 = f"/Users/manansuri/Desktop/Research/btp/ijcnlp_dailydialog/{split}/dialogues_act_{split}.txt"
# out_file_1 = f'/Users/manansuri/Desktop/Research/btp/sample/dialogue_act_filter.txt'

# in_file_2 = f"/Users/manansuri/Desktop/Research/btp/ijcnlp_dailydialog/{split}/dialogues_emotion_{split}.txt"
# out_file_2 = f'/Users/manansuri/Desktop/Research/btp/sample/dialogue_emotion_filter.txt'

# txt_file_path_in = f'/Users/manansuri/Desktop/Research/btp/{split}/dailydialog_filter.txt'

filter_lines_by_threshold(csv_file_path, in_file_1, out_file_1)
# filter_lines_by_threshold(csv_file_path, in_file_2, out_file_2)
