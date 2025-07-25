<div align="center">

# Colab-IgGM: An Interactive Workflow for Antibody and Nanobody Design

[![Notebook](http://img.shields.io/badge/Notebook-GoogleColab-orange.svg)](https://colab.research.google.com/drive/1-v8-anrA5rtlZzqT-g7HtVLzK1FZQzLv?usp=sharing)
[![iGEM Team](https://img.shields.io/badge/iGEM-UNILA--LatAm-blueviolet.svg)](https://www.instagram.com/igem_synfronteras/)

<img src="https://github.com/Lefrunila/Colab-IgGM/blob/master/ezgif-3bc0745dd178ef.gif" alt="IgGM Animation" height="350">

</div>

---

## Overview

**Colab-IgGM** provides a user-friendly and interactive pipeline for the design of novel antibodies and nanobodies against a specific antigen epitope. This notebook is a powerful front-end for the deep learning model **IgGM**, streamlining the entire process from preparing input files to analyzing and visualizing the final designs.

This workflow was created to support the **NANODEN** project by the **iGEM UNILA-LatAm team** for the 2025 competition. It integrates several state-of-the-art bioinformatics tools into a series of easy-to-use, sequential cells.

### Key Tools Integrated
* **Design Engine**: [IgGM](https://www.biorxiv.org/content/10.1101/2024.09.19.613838v2) for generative antibody design.
* **Chain Annotation**: [ANARCI](https://github.com/oxpig/ANARCI?tab=readme-ov-file) for identifying VH/VL/VHH chains, numbering residues Identifying CDRs and masking them for redesign.
* **Epitope Prediction**: [SEPPA 3.0](http://www.badd-cao.net/seppa3/) for predicting epitopes on an antigen surface.
* **Structure Visualization**: [py3Dmol](https://3dmol.csb.pitt.edu/) for interactive 3D visualization of protein structures.
* **(Optional) Structure Relaxation**: [PyRosetta](https://www.pyrosetta.org/) for energy minimization of the final designs. This was already integrated in the original IgGM.

---

## Workflow

The notebook is designed as a modular, step-by-step pipeline. The general workflow is as follows:

PDB Input ➔ 1. Clean PDB (Remove non-protein) ➔ 2. Analyze Chains (Identify VH/VL/VHH) ➔ 3. Prepare Design Files (Mask CDRs, Merge Antigen, Rename Chains) ➔ 4. Define Epitope ➔ 5. Run IgGM Design ➔ 6. Visualize & Align Results

---

## How to Use

To get started, open the [notebook](https://colab.research.google.com/drive/1-v8-anrA5rtlZzqT-g7HtVLzK1FZQzLv?usp=sharing) in Google Colab. The notebook is divided into numbered cells that should be run in order. Detailed instructions are provided in each cell's title and markdown text, also at the bottom of the notebook.

### **Part 1: Setup & Pre-processing**
1.  **Install Dependencies:** Run the first few cells to install all necessary libraries and tools like `condacolab`, `pyrosettacolabsetup`, `ANARCI`, and `HMMER`. This may require restarting the Colab session once.
2.  **Create Input Folder:** The notebook will automatically create the necessary directory structure, including `/content/IgGM/inputs`.

### **Part 2: PDB Pre-processing**
1.  **Clean PDB:** Upload your PDB file. The script will remove all non-protein atoms (water, ligands, ions) and create a `cleaned_` version of your file, displaying both structures for comparison.
2.  **(Optional) Remove Chains:** If your antigen has multiple chains and some are not part of the epitope, you can remove them to simplify the structure.
3.  **Rename Chains:** For the design step, IgGM requires the antibody heavy chain to be named `H` and the light chain `L`. Use this cell to rename the chains identified by ANARCI.

### **Part 3: Epitope & Design File Generation**
1.  **Identify Antibody Chains:** The notebook uses **ANARCI** to analyze your cleaned PDB, identify which chains are VH, VL, or VHH, and saves the sequences to a FASTA file.
2.  **Mask CDRs for Redesign:** A new FASTA file is created where the CDRs of the identified antibody chains are masked with `X`'s. This file tells IgGM which regions to redesign.
3.  **Merge Antigen Chains:** IgGM requires the antigen to be a single chain. This cell merges all antigen chains into a new chain `A`.
4.  **Define Epitope:** You have two options:
    * **From Complex (Recommended):** If you started with an antibody-antigen complex, the script can automatically identify the epitope residues.
    * **From Antigen Only:** If you only have an antigen structure, follow the instructions to use the **SEPPA 3.0** server. Then, upload the results to the notebook, which will cluster the predicted residues into epitope "hotspots" for IgGM.

### **Part 4: Antibody Design and Visualization**
1.  **Run IgGM Design:** This is the core step. Fill in the paths to your prepared FASTA and PDB files, select an epitope, and adjust the design parameters (number of samples, relaxation, etc.). Run the cell to start the design process.
2.  **Visualize Results:** After the design is complete, use this cell to compare your original input structure with one of the new designs. It will automatically align the structures, calculate the RMSD, and provide three viewers: input, output, and an overlapped comparison.

---

### Future Work and Potential

The current workflow successfully implements a powerful pipeline for antibody analysis and design preparation using **IgGM**. The next frontier is to integrate the complete, end-to-end functional engineering capabilities demonstrated in the groundbreaking work by Yu Kong, Jiale Shi, Fandi Wu, and colleagues.

Their paper introduces [**TFDesign-sdAb**](https://www.biorxiv.org/content/10.1101/2025.05.09.653014v1), a framework whose power comes from the synergy of two novel components working together: a specialized generator and an expert ranker. The key innovations that make this approach so successful are:

* **A Re-architected IgGM for Framework Design:** The authors intelligently modified the IgGM model with a two-phase training strategy. This allows it to optimize not just the CDRs, but also the crucial **Framework Regions (FRs)**, which is essential for engineering new functions like Protein A binding while preserving the antibody's original antigen affinity.

* **A Fine-Tuned A2binder for Accurate Ranking:** A generative model can create thousands of candidates, but this is only useful if the best ones can be identified. The authors demonstrated that a generic affinity model failed on sdAbs, but after **fine-tuning A2binder with a small, specific set of sdAb affinity data**, its performance improved dramatically. This expertly adapted ranker is the critical component that filters the vast design space to find the candidates with the highest chance of success.

The authors' amazing work in combining these innovations led to a remarkable **100% success rate in their specific task**: engineering Protein A binding into sdAbs that previously could not. If the repository containing the specialized **IgGM-FR** model and, most importantly, the **fine-tuned A2binder model** with its weights were made public, this Colab notebook could be transformed into a complete, all-in-one tool. It would empower researchers to go directly from an antigen structure to a small, highly enriched set of computationally validated antibody and nanobody designs that are truly ready for wet-lab synthesis and testing, massively accelerating the pace of therapeutic discovery.

---

### **License Information**

The original code in this Colab notebook, created by the author, is licensed under the Apache 2.0 License.

This notebook is a fork of and interacts with the IgGM repository. The underlying IgGM source code is provided under the [**PolyForm Noncommercial License 1.0.0**](https://polyformproject.org/licenses/noncommercial/1.0.0), making it free for non-commercial, academic, and personal research purposes. This notebook also integrates several external tools and libraries, each with its own licensing terms that must be respected:

* **SEPPA 3.0**: The web server is provided for **academic use only**. Commercial applications require contacting the authors at `zwcao@fudan.edu.cn`.
* **PyRosetta**: The PyRosetta software ("Software") has been developed by the contributing researchers and institutions of the Rosetta Commons ("Developers") and made available through the University of Washington ("UW") for noncommercial, non-profit use. For more information about the Rosetta Commons, please see www.rosettacommons.org [**www.rosettacommons.org**](https://www.rosettacommons.org ). If you wish to use the Software for any commercial purposes, including fee- based service projects, you will need to execute a separate licensing agreement with the UW and pay a fee. In that case, please contact: license@uw.edu 
* **ANARCI**: Provided under the permissive [**BSD 3-Clause License**](https://opensource.org/licenses/BSD-3-Clause).
* **HMMER**: Available under the [**GNU GPLv3**](https://www.gnu.org/licenses/gpl-3.0.html).
* **PyTorch Geometric (PyG)**: The library and its components (`torch_scatter`, `torch_sparse`, etc.) are provided under the [**MIT License**](https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE).
* **PyTorch**: The core deep learning framework is licensed under a permissive [**BSD-style License**](https://github.com/pytorch/pytorch/blob/main/LICENSE).
* **Biopython**: Distributed under the liberal [**Biopython License**](https://biopython.org/wiki/License/).
* **OpenMM**: Provided under the [**MIT License**](https://github.com/openmm/openmm/blob/main/LICENSE.txt).
* **PDBFixer**: Provided under the [**MIT License**](https://github.com/openmm/pdbfixer/blob/main/LICENSE).
* **py3Dmol**: Provided under the [**MIT License**](https://pypi.org/project/py3Dmol/).
* **Condacolab**: Provided under the [**MIT License**](https://github.com/conda-incubator/condacolab/blob/main/LICENSE).

### **Citations & Acknowledgments**

Please cite the relevant authors and tools in any work that uses this notebook.

**Primary Publications:**
* **For IgGM:**
    > Wang, Rubo et al. "IgGM: A Generative Model for Functional Antibody and Nanobody Design." *The Thirteenth International Conference on Learning Representations*, 2025.
* **For ANARCI:**
    > Dunbar, J. & Deane, C. M. "ANARCI: antigen receptor numbering and receptor classification." *Bioinformatics*, 32(2), 298–300 (2016).
* **For SEPPA 3.0:**
    > Zhou, C., Chen, Z., Zhang, L. et al. "SEPPA 3.0—enhanced spatial epitope prediction enabling glycoprotein antigens." *Nucleic Acids Research*, 47(W1), W388–W394 (2019).
* **For PyRosetta:**
    > Chaudhury, S., Lyskov, S. & Gray, J. J. "PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta." *Bioinformatics*, 26(5), 689-691 (2010).

**Acknowledgments:**
* We thank the **IgGM team** for developing their excellent model and making the code available for non-commercial research.
* We thank the **Oxford Protein Informatics Group (OPIG)** for creating the indispensable **ANARCI** tool and the **Cao Lab** for developing and maintaining the **SEPPA 3.0** server.
* A special thanks to **David Koes** for his awesome **py3Dmol** plugin, which makes the interactive visualizations in this notebook possible.
* This Colab notebook was created and adapted by **Luis Eduardo Figueroa** (`lef.rivera.2021@aluno.unila.edu.br`) for the **NANODEN** project of the **iGEM UNILA-LatAm team**. Follow our progress on Instagram: **[@igem_synfronteras](https://www.instagram.com/igem_synfronteras/)**.


