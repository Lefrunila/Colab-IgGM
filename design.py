# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys
import torch
import tqdm

# Added crop_sequence_with_epitope import
from IgGM.protein import cal_ppi, crop_sequence_with_epitope

# Ensure IgGM is in the Python path
sys.path.append('/content/IgGM')

from IgGM.deploy import AbDesigner
from IgGM.utils import setup
from IgGM.protein.parser import parse_fasta, PdbParser
from IgGM.model.pretrain import esm_ppi_650m_ab, antibody_design_trunk, IGSO3Buffer_trunk


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody sequence and structure co-design w/ IgGM')
    parser.add_argument('--fasta', '-f', type=str, required=True, help='Path to input antibody FASTA file')
    parser.add_argument('--antigen', '-ag', type=str, required=True, help='Path to input antigen PDB file')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory for PDB files')
    parser.add_argument('--epitope', default=None, nargs='+', type=int, help='Epitope residues in antigen chain A')
    parser.add_argument('--device', '-d', type=str, default=None, help='Inference device')
    parser.add_argument('--steps', '-s', type=int, default=10, help='Number of sampling steps')
    parser.add_argument('--chunk_size', '-cs', type=int, default=64, help='Chunk size for long-chain inference')
    parser.add_argument('--num_samples', '-ns', type=int, default=1, help='Number of samples for each input')
    parser.add_argument(
        '--relax', '-r',
        action='store_true',
        help='relax structures after design',
    )
    # Added --cal_epitope flag
    parser.add_argument(
        '--cal_epitope', '-ce',
        action='store_true',
        default=False,
        help='if use, will calculate epitope from antigen pdb and exit',
    )
    # Added --max_antigen_size flag
    parser.add_argument(
        '--max_antigen_size', '-mas',
        type=int,
        default=2000,
        help='max size of antigen chain, default is 2000',
    )

    return parser.parse_args()


def predict(args):
    """Predict antibody & antigen sequence and structures w/ pre-trained IgGM-Ag models."""
    fasta_path = args.fasta
    pdb_path = args.antigen

    sequences, ids, _ = parse_fasta(fasta_path)
    assert len(sequences) in (1, 2, 3), "FASTA file must contain 1, 2, or 3 chains"
    
    # Added logic for --cal_epitope
    if args.cal_epitope:
        print("Calculating epitope only...")
        epitope = cal_ppi(pdb_path, ids)
        epitope_indices = torch.nonzero(epitope).flatten().tolist()
        print(f"Calculated epitope indices: {' '.join(str(i) for i in epitope_indices)}")
        return # Exit after calculating

    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids) if seq_id != ids[-1]]
    
    _, basename = os.path.split(fasta_path)
    name = basename.split(".")[0]

    # Load antigen structure
    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=ids[-1])

    if args.epitope is None:
        try:
            epitope = cal_ppi(pdb_path, ids)
        except Exception:
            epitope = args.epitope
    else:
        epitope = torch.zeros(len(aa_seq))
        for i in args.epitope:
            if i < len(aa_seq):
                epitope[i] = 1
            else:
                print(f"Warning: Epitope index {i} is out of range.")

    # Added logic for --max_antigen_size
    if len(aa_seq) > args.max_antigen_size:
        print(f"Antigen length ({len(aa_seq)}) exceeds max size ({args.max_antigen_size}). Cropping...")
        aa_seq, atom_cord, atom_cmsk, epitope, _ = crop_sequence_with_epitope(
            aa_seq, atom_cord, atom_cmsk, epitope, max_len=args.max_antigen_size
        )

    chains.append({
        "sequence": aa_seq,
        "cord": atom_cord,
        "cmsk": atom_cmsk,
        "epitope": epitope,
        "id": "A"
    })

    batches = [
        {
            "name": name,
            "chains": chains,
            "output": f"{args.output}/{name}_{i}.pdb",
        }
        for i in range(args.num_samples)
    ]

    # Free GPU memory before execution
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

    # Reduce memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load IgGM model
    designer = AbDesigner(
        ppi_path=esm_ppi_650m_ab(),
        design_path=antibody_design_trunk(),
        buffer_path=IGSO3Buffer_trunk(),
        config=args,
    )
    designer.to(device)

    # Move specific layers to CPU to reduce memory usage
    try:
        designer.model.encoder.to("cpu")
        print("Moved encoder to CPU.")
    except AttributeError:
        print("Warning: Unable to move some layers to CPU. Check layer names.")

    chunk_size = args.chunk_size
    print(f"# Inference samples: {len(batches)}")

    for task in tqdm.tqdm(batches):
        designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size, relax=args.relax)


def main():
    args = parse_args()
    setup(True)
    predict(args)


if __name__ == "__main__":
    main()
