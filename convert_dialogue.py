def process_dialogues(input_file, output_file):
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            dialogues = line.strip().split('__eou__')  # Split the dialogues based on the __eou__ token
            output = "#ROOT# "
            speaker = ''
            for dialogue in dialogues:
                if dialogue.strip() != '':
                    output += f"<utterance> {dialogue.strip()} "
                    speaker = ''  # Alternate between speakers
            output += "\n"
            output_f.write(output)

# File paths
input_file = '/Users/manansuri/Desktop/Research/btp/ijcnlp_dailydialog/train/dialogues_train.txt'
output_file = '/Users/manansuri/Desktop/Research/btp/ddp_format/dialogues_train.txt'

# Process the dialogues
process_dialogues(input_file, output_file)
