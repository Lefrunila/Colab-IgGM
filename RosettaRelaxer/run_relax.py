#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025, [Your Name or Alias]. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import subprocess
import sys
from Bio.PDB import PDBParser

# (The rest of the script is the same as the pre_relax.py we designed before)

def find_rosetta_executable():
    """Find the Rosetta scripts executable, checking common paths."""
    rosetta_home = os.environ.get("ROSETTA_HOME")
    if rosetta_home:
        path = os.path.join(rosetta_home, "main/source/bin/rosetta_scripts.default.linuxgccrelease")
        if os.path.exists(path):
            return path
    print("Error: Rosetta executable not found. Please set the ROSETTA_HOME environment variable.", file=sys.stderr)
    return None

def get_chain_ids(pdb_path):
    """Parses a PDB file to get the IDs of the first two chains."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model = structure[0]
    chain_ids = [chain.id for chain in model]
    if len(chain_ids) < 2:
        print(f"Error: Expected at least 2 chains in {pdb_path}, but found {len(chain_ids)}.", file=sys.stderr)
        return None
    return f"{chain_ids[0]}_{chain_ids[1]}"

def create_relax_protocol(xml_path, partners):
    """Creates a RosettaScripts XML file for relaxing a protein complex."""
    xml_content = f"""
<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="ref2015" weights="ref2015_cart"/>
    </SCOREFXNS>
    <MOVERS>
        <DockingProtocol name="relax_dock" partners="{partners}" scorefxn="ref2015" fullatom="1" ignore_default_docking_task="1" docking_local_refine="1"/>
    </MOVERS>
    <PROTOCOLS>
        <Add mover_name="relax_dock"/>
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    print(f"Generated Rosetta protocol at: {xml_path}")

def run_rosetta_relax(rosetta_path, input_pdb, output_dir, xml_path):
    """Constructs and runs the Rosetta command line for relaxation."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    command = [
        rosetta_path, "-s", input_pdb, "-parser:protocol", xml_path,
        "-out:path:pdb", output_dir, "-out:path:score", output_dir,
        "-out:file:scorefile", "rosetta_scores.sc", "-nstruct", "1",
        "-ignore_unrecognized_res", "-ex1", "-ex2", "-overwrite"
    ]
    print("\nRunning Rosetta Command:\n" + " ".join(command))
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print("\nRosetta relaxation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Rosetta Failed ---", file=sys.stderr)
        print(f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='A standalone tool to run Rosetta relaxation on a PDB complex.')
    parser.add_argument('--input_pdb', type=str, required=True, help='Path to the input PDB file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save relaxed PDB and score files.')
    parser.add_argument('--rosetta_path', type=str, help='Optional: Path to rosetta_scripts. If not set, checks ROSETTA_HOME.')
    args = parser.parse_args()

    rosetta_exec = args.rosetta_path or find_rosetta_executable()
    if not rosetta_exec: sys.exit(1)

    partners = get_chain_ids(args.input_pdb)
    if not partners: sys.exit(1)
    
    print(f"Detected docking partners: {partners}")
    xml_path = os.path.join(args.output_dir, "relax_protocol.xml")
    create_relax_protocol(xml_path, partners)
    run_rosetta_relax(rosetta_exec, args.input_pdb, args.output_dir, xml_path)

if __name__ == '__main__':
    main()
