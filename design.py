# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys
import torch
import tqdm
import gc
import subprocess # Use subprocess for better output control

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
    parser.add_argument(
        '--cal_epitope', '-ce',
        action='store_true',
        default=False,
        help='if use, will calculate epitope from antigen pdb and exit',
    )
    parser.add_argument(
        '--max_antigen_size', '-mas',
        type=int,
        default=2000,
        help='max size of antigen chain, default is 2000',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Set a specific random seed for reproducibility. If not set, a random seed will be used.'
    )
    return parser.parse_args()


def predict(args):
    """
    Manages the generation of multiple samples by calling the main script
    as a subprocess for each sample to ensure memory is cleared.
    """
    if args.cal_epitope:
        # For epitope calculation, we can run it directly as it's a quick, one-off task
        print("Calculating epitope only...")
        sequences, ids, _ = parse_fasta(args.fasta)
        epitope = cal_ppi(args.antigen, ids)
        epitope_indices = torch.nonzero(epitope).flatten().tolist()
        print(f"epitope: {' '.join(str(i) for i in epitope_indices)}")
        return

    print(f"# Running {args.num_samples} samples, each in a separate process to manage memory...")

    # The main script's path is this file itself
    script_path = os.path.abspath(__file__)

    for i in range(args.num_samples):
        print(f"\n--- Starting Sample {i+1}/{args.num_samples} ---")
        
        # Construct the command to call this script for a single sample
        # We pass all the original arguments, but set num_samples to 1
        # and adjust the seed for variety
        output_file = f"{args.output}/{os.path.splitext(os.path.basename(args.fasta))[0]}_{i}.pdb"
        
        cmd = [
            sys.executable, script_path,
            '--fasta', args.fasta,
            '--antigen', args.antigen,
            '--output', os.path.dirname(output_file),
            '--num_samples', '1', # Run one sample at a time
            '--steps', str(args.steps),
            '--chunk_size', str(args.chunk_size),
            '--max_antigen_size', str(args.max_antigen_size),
        ]
        
        # Add optional arguments
        if args.epitope:
            cmd.extend(['--epitope'] + [str(e) for e in args.epitope])
        if args.relax:
            cmd.append('--relax')
        if args.seed is not None:
            cmd.extend(['--seed', str(args.seed + i)])
            
        # This is a dummy argument to rename the single output file correctly
        cmd.extend(['--_internal_output_name', os.path.basename(output_file)])

        # Use subprocess.Popen to stream output live
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            print(f"--- Sample {i+1} completed successfully ---")
        else:
            print(f"--- ⚠️ Sample {i+1} failed ---")


def single_sample_inference(args):
    """
    This function contains the logic to generate a single antibody.
    It is called by the main script when run as a subprocess.
    """
    # This logic is mostly the same as the original predict function
    # but simplified for a single run.
    sequences, ids, _ = parse_fasta(args.fasta)
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids) if seq_id != ids[-1]]
    
    name = os.path.splitext(os.path.basename(args.fasta))[0]
    output_filename = args._internal_output_name # Use the internal name

    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(args.antigen, chain_id=ids[-1])

    if args.epitope is None:
        try: epitope = cal_ppi(args.antigen, ids)
        except: epitope = None
    else:
        epitope = torch.zeros(len(aa_seq))
        for i in args.epitope:
            if i < len(aa_seq): epitope[i] = 1

    if len(aa_seq) > args.max_antigen_size:
        aa_seq, atom_cord, atom_cmsk, epitope, _ = crop_sequence_with_epitope(
            aa_seq, atom_cord, atom_cmsk, epitope, max_len=args.max_antigen_size
        )

    chains.append({"sequence": aa_seq, "cord": atom_cord, "cmsk": atom_cmsk, "epitope": epitope, "id": "A"})
    
    task = {"chains": chains, "output": os.path.join(args.output, output_filename)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    designer = AbDesigner(
        ppi_path=esm_ppi_650m_ab(), design_path=antibody_design_trunk(),
        buffer_path=IGSO3Buffer_trunk(), config=args,
    )
    designer.to(device)

    with torch.no_grad():
        # tqdm is used here to show the progress for the single sample
        for _ in tqdm.tqdm(range(1), desc=f"Designing {name}"):
            designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=args.chunk_size, relax=args.relax)

def main():
    parser = parse_args()
    # Add a temporary argument to handle file naming in subprocesses
    parser.add_argument('--_internal_output_name', type=str, default=None)
    args = parser.parse_args()

    # The script decides its role based on whether it's called with the internal flag
    if args._internal_output_name:
        # This is a worker process, run a single sample
        setup(True, seed=args.seed)
        single_sample_inference(args)
    else:
        # This is the main process, manage the workers
        setup(True, seed=args.seed)
        predict(args)

if __name__ == "__main__":
    main()
