import torch

integerized_samples = ['1234567', 'ABDNCJJLSGASYA', 'ABDNCJJLSGVVV']
integerized_seqs = [s[0] for s in integerized_samples]
print(integerized_seqs)
maxlen = ([len(seq) for seq in integerized_seqs])
print(maxlen)
