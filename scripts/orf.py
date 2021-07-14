from Bio import SeqIO
import re
import torch

H_train = 'H_train.fasta'

# define a dic
letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb because the padded_input_tensor is zero


# TODO input should be either string or sequence of ints
def orf_finder(file_name):
    ORF = []
    label_orf = []
    with open(file_name) as fn:
        for record in SeqIO.parse(fn, 'fasta'):
            stop_to_stop_codons = re.split(r'TAA|TAG|TGA', str(record.seq))
            # print(stop_to_stop_codons)
            codon_len = [len(seq) for seq in stop_to_stop_codons]
            codon_len = codon_len[1:-2]
            # print(codon_len)
            for loop in codon_len:
                index_max = codon_len.index(max(codon_len))
                if (max(codon_len) % 3 == 0) & (max(codon_len) != 0):
                    orf = stop_to_stop_codons[index_max + 1]
                    # print(orf)
                    break
                codon_len[index_max] = 0
                orf = '0'
                # print(codon_len[index_max])
            ORF.append(orf)
            orfs = torch.tensor(ORF, dtype=torch.long)

            label = 0 if 'CDS' in record.id else 1
            label_orf.append(label)
            labels_orf = torch.tensor(label_orf, dtype=torch.long)

        return orfs, labels_orf  # TODO remove labeling


[orfs, labels_orf] = orf_finder(H_train)
