'''
import torch
import torch.nn.functional as nn

input_tensor = torch.rand(296, 4, 6)
hidden_tensor = torch.rand(4, 128)

combined = torch.cat((input_tensor[0], hidden_tensor), -1)
print(combined.size())

print(combined)


## nn.Linear
import torch
import torch.nn as nn
m = nn.Linear(20, 30)
print(m)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
torch.Size([128, 30])


import torch
import torch.nn as nn
m = torch.nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()


# input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
print(input)
print(input.size())
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
print(target)
print(target.size())
aa = m(input)
output = loss(aa, target)
print(aa.size())
print(output)
output.backward()


'''
'''
## ORF Indicating
for index, record in enumerate(SeqIO.parse(open("file.fasta"), "fasta")):
    for strand, nuc in [(+1, record.seq), (-1, record.seq.reverse_complement())]:
        for frame in range(3):
            length = 3 * ((len(record)-frame) // 3) #Multiple of three
        for orf in DNA:
            orf = re.search(r"ATG(?:(?!TAA|TAG|TGA)...)*(?:TAA|TAG|TGA)", DNA).group()
            #print "ID = %s, length %i, frame %i, strand %i" \
                  #% record.id, len(orf), frame, strand)
            
            
'''
'''
import re
#seq = [TGGTAACAATAAGATCTGTGGTTGGAATTATGAATGTCCAAAGTTTGAAGAGGATGTTTTGAGCAGTGACATTATAATTCTGACAATAACACGATGCATAGCCATCCTGTATATTTACTTCCAGTTCCAGAATTTACGTCAACTTGGATCAAAATATATTTTGGGTATTGCTGGCCTTTTCACAATTTTCTCAAGTTTTGTATTCAGTACAGTTGTCATT]
#seq = ['T''G''G''T''A''A','C','A','A','T','A','A','G','A','T','C','T','G','T','G','G','T','T','G','G','A','A','T','T','A','T','G','A']
seq = 'ATTTTTTTCCCCCCCCCGGGGGGGGG'

codon = re.split(r'TAA|TAG|TGA', str(seq))
print(codon)
print(len(codon))
codon_len = [len(seq) for seq in codon]
codon_len = codon_len[1:-1]
print(codon_len)
for index in codon_len:
    print(index)
    index = codon_len.index(max(codon_len))
    if max(codon_len) % 3 == 0:
        orf = codon[index + 1]
        print(orf)
        break
    else:
        codon_len[index] = 0
        print(codon_len[index])
'''
'''
def orf_finder(file_name):
    ORF = []
    orf_length = []
    orf_ratio = []
    orf_targets = []
    orf_cRNA = []
    orf_ncRNA = []
    orf_cRNA_length = []
    orf_ncRNA_length = []
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
            # orf = torch.tensor(orf, dtype=torch.long)
            ORF.append(orf)
            orf_length.append(codon_len[index_max])
            orf_ratio.append(codon_len[index_max]/len(record.seq))
            #label = 0 if 'CDS' in record.id else 1
            if 'CDS' in record.id:
                label = 0
                orfc = orf
                orfc_length = codon_len[index_max]
            else:
                label = 1
                orfnc = orf
                orfnc_length = codon_len[index_max]
            orf_ncRNA.append(orfnc)
            orf_ncRNA_length.append(orfnc_length)

            orf_targets.append(label)
            # orf_targets = torch.tensor(label, dtype=torch.long)
        orf_lst = [ORF, orf_targets, orf_length, orf_ratio, orf_cRNA, orf_ncRNA, orf_cRNA_length, orf_ncRNA_length]
        return orf_lst


[ORF, orf_targets, orf_length, orf_ratio, orf_cRNA, orf_ncRNA, orf_cRNA_length, orf_ncRNA_length] = orf_finder(H_train)

# the histogram of the orf
n, bins, patches = plt.hist(orf_cRNA_length, 50, density=True, facecolor='g', alpha=0.75)

plt.xlabel('ORF length')
plt.ylabel('number')
plt.title('length distribution of ORF for ncRNA')
plt.grid(True)
plt.show()

'''

'''
import torch
import torch.nn as nn
rnn = nn.RNN(10, 20, 2)
print(rnn)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)

print(output)
'''
import torch
from torch.nn.utils.rnn import pack_sequence
a = torch.tensor([1,2,3])
b = torch.tensor([4,5])
c = torch.tensor([6])
print(pack_sequence([a, b, c]))

















