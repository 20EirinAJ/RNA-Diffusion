from Bio import SeqIO

with open("A_4R.fastq", "r") as fastq_file, open("A_4R.fasta", "w") as fasta_file:
    sequences = SeqIO.parse(fastq_file, "fastq")
    SeqIO.write(sequences, fasta_file, "fasta")
