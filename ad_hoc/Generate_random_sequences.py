import random

def generate_dna_sequence(length):
    """Generate a random DNA sequence of a given length."""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(bases) for _ in range(length))

def generate_multiple_dna_sequences(num_sequences, sequence_length):
    """Generate multiple DNA sequences."""
    return [generate_dna_sequence(sequence_length) for _ in range(num_sequences)]

# Generating 10000 DNA sequences each of length 147
num_sequences = 10000
sequence_length = 147
# dna_sequences = generate_multiple_dna_sequences(num_sequences, sequence_length)

# Writing the DNA sequences to a FASTA file
with open('./Data/random_dna_sequences.fasta', 'w') as file:
    for i in range(num_sequences):
        sequence = generate_dna_sequence(sequence_length)
        file.write(f">Sequence_{i+1}\n{sequence}\n")

# The file has been saved as 'random_dna_sequences.fasta'
# fasta_file_path = '/mnt/data/dna_sequences.fasta'
