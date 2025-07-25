#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.

import argparse
import gzip
import logging
import os
import sys
import uuid
import warnings
from collections import OrderedDict

import torch
import tqdm

# Biopython imports for structure parsing and FASTA handling
from Bio import BiopythonWarning
from Bio.PDB import PDBParser as BioPDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# ------------------------------------------------------------------
# Minimal constant definitions
RESD_NAMES_3C = {"ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
                 "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
                 "THR", "VAL", "TRP", "TYR"}

RESD_MAP_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

RESD_MAP_1TO3 = {v: k for k, v in RESD_MAP_3TO1.items()}

N_ATOMS_PER_RESD = 14

ATOM_NAMES_PER_RESD = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'], 'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'], 'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'], 'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CD1', 'CG2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'], 'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'], 'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'], 'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], 'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'], 'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CE1', 'OH', 'CZ', 'CE2', 'CD2'], 'UNK': ['N', 'CA', 'C', 'O']
}

class PdbParseError(Exception): pass

class PdbParser:
    @classmethod
    def load(cls, pdb_fpath, chain_id=None):
        try:
            structure = cls.__get_structure(pdb_fpath)
            chain = cls.__get_chain(structure, None, chain_id)
            # Now returns residue IDs along with other data
            aa_seq, atom_cords, atom_masks, res_ids = cls.__get_atoms(chain)
            return aa_seq, atom_cords, atom_masks, res_ids, None
        except PdbParseError as e:
            return None, None, None, None, e

    @classmethod
    def save_multimer(cls, prot_data, path):
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w', encoding='UTF-8') as o_file:
            idx_atom_base = 0
            for chain_id, chain_data in prot_data.items():
                pdb_strs = cls.__get_pdb_strs(
                    chain_data['seq'], chain_id, chain_data['cord'], 
                    chain_data['cmsk'], chain_data['res_ids'], idx_atom_base
                )
                o_file.write('\n'.join(pdb_strs) + '\n')
                idx_atom_base += int(torch.sum(chain_data['cmsk']).item())

    @classmethod
    def __get_pdb_strs(cls, aa_seq, chain_id, atom_cords, atom_masks, res_ids, idx_atom_base=0):
        pdb_strs = []
        n_atoms = 0
        last_resd_name_3c, last_res_id = "UNK", -1
        clean_chain_id = str(chain_id).strip()[0] if str(chain_id).strip() else ' '

        for i, resd_name_1c in enumerate(aa_seq):
            if torch.sum(atom_masks[i]) == 0: continue
            
            res_id = res_ids[i]
            last_res_id = res_id
            resd_name_3c = RESD_MAP_1TO3.get(resd_name_1c, "UNK")
            last_resd_name_3c = resd_name_3c
            atom_names = ATOM_NAMES_PER_RESD.get(resd_name_3c, ATOM_NAMES_PER_RESD['UNK'])

            for idx_atom, atom_name in enumerate(atom_names):
                if idx_atom >= atom_masks.shape[1] or atom_masks[i, idx_atom] == 0: continue
                n_atoms += 1
                serial = n_atoms + idx_atom_base
                x, y, z = atom_cords[i, idx_atom]
                
                atom_name_padded, alt_loc = f'{atom_name:<4}', ' '
                
                line = (f"ATOM  {serial:5d} {atom_name_padded}{alt_loc}{resd_name_3c:>3} {clean_chain_id:1}"
                        f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2}")
                pdb_strs.append(line)
        
        if n_atoms > 0:
            ter_serial = n_atoms + idx_atom_base + 1
            pdb_strs.append(f"TER   {ter_serial:5d}      {last_resd_name_3c:>3} {clean_chain_id:1}{last_res_id:4d}")
        
        return pdb_strs

    @classmethod
    def __get_structure(cls, path):
        parser = BioPDBParser(QUIET=True)
        try:
            handle = gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r', encoding='UTF-8')
            with handle:
                return parser.get_structure(str(uuid.uuid4()), handle)
        except Exception as e:
            raise PdbParseError(f'Biopython failed to parse: {e}') from e

    @classmethod
    def __get_chain(cls, structure, model_id, chain_id):
        try:
            model = structure[model_id or 0]
            return model[chain_id] if chain_id is not None else next(model.get_chains())
        except (KeyError, StopIteration) as e:
            raise PdbParseError('Chain not found') from e

    @classmethod
    def __get_atoms(cls, chain):
        residues = [res for res in chain if res.get_id()[0] == ' ' and res.get_resname() in RESD_MAP_3TO1]
        
        seq_len = len(residues)
        aa_seq = ''.join([RESD_MAP_3TO1.get(res.get_resname()) for res in residues])
        res_ids = [res.get_id()[1] for res in residues]
        atom_cords = torch.zeros((seq_len, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
        atom_masks = torch.zeros((seq_len, N_ATOMS_PER_RESD), dtype=torch.int8)

        for i, residue in enumerate(residues):
            resd_name = residue.get_resname()
            atom_names = ATOM_NAMES_PER_RESD[resd_name]
            for idx_atom, atom_name in enumerate(atom_names):
                if residue.has_id(atom_name):
                    atom_cords[i, idx_atom] = torch.tensor(residue[atom_name].get_coord())
                    atom_masks[i, idx_atom] = 1
        return aa_seq, atom_cords, atom_masks, res_ids

def export_fasta(sequences, ids, output):
    records = [SeqRecord(Seq(seq), id=id, description="") for seq, id in zip(sequences, ids)]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    SeqIO.write(records, output, "fasta")

def main():
    parser = argparse.ArgumentParser(description='Merge PDB chains.')
    parser.add_argument('--antigen', type=str, required=True, help='Input PDB file path.')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory path.')
    parser.add_argument('--antibody_ids', type=str, default="H_L", help='Antibody chains to keep, e.g., H_L.')
    parser.add_argument('--merge_ids', type=str, required=True, help='Chains to merge, e.g., A_B.')
    args = parser.parse_args()

    antibody_ids = args.antibody_ids.split('_') if args.antibody_ids and args.antibody_ids.lower() != "none" else []
    merge_ids = args.merge_ids.split('_') if args.merge_ids else []
    chains_data = OrderedDict()
    all_seqs, all_ids = [], []

    for chn_id in antibody_ids:
        seq, cords, masks, res_ids, err = PdbParser.load(args.antigen, chain_id=chn_id)
        if err: print(f"Warning: Skipping antibody chain {chn_id}: {err}"); continue
        chains_data[chn_id] = {'seq': seq, 'cord': cords, 'cmsk': masks, 'res_ids': res_ids}
        all_seqs.append(seq)
        all_ids.append(chn_id)

    merged_seq, merged_cords, merged_masks, merged_res_ids = "", [], [], []
    residue_offset = 0
    for chn_id in merge_ids:
        seq, cords, masks, res_ids, err = PdbParser.load(args.antigen, chain_id=chn_id)
        if err: print(f"Warning: Skipping merge chain {chn_id}: {err}"); continue
        
        if merged_res_ids:
            residue_offset = max(merged_res_ids) + 10 # Add gap of 10
        
        offset_res_ids = [r_id + residue_offset for r_id in res_ids]

        merged_seq += seq
        merged_cords.append(cords)
        merged_masks.append(masks)
        merged_res_ids.extend(offset_res_ids)

    if merged_seq:
        new_chain_id = merge_ids[0] if merge_ids else 'A'
        chains_data[new_chain_id] = {
            'seq': merged_seq,
            'cord': torch.cat(merged_cords, dim=0),
            'cmsk': torch.cat(merged_masks, dim=0),
            'res_ids': merged_res_ids
        }
        all_seqs.append(merged_seq)
        all_ids.append(new_chain_id)

    if not chains_data: print("Error: No valid chains processed."); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.basename(args.antigen).rsplit('.', 1)[0]
    pdb_output = os.path.join(args.output, f"{base_name}_merge.pdb")
    fasta_output = os.path.join(args.output, f"{base_name}_merge.fasta")
    
    PdbParser.save_multimer(chains_data, pdb_output)
    export_fasta(all_seqs, all_ids, fasta_output)
    
    print(f'Merged PDB saved to {pdb_output}')
    print(f'FASTA file saved to {fasta_output}')

if __name__ == '__main__':
    main()
