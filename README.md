# HEP-EEG Toolbox

A lightweight EEG preprocessing and analysis toolbox designed for fast, flexible,
and researcher-friendly workflows.  
The toolbox provides simple interfaces for:

- Loading XDF recordings  
- Preprocessing raw EEG / EOG / ECG / RESP  
- Event extraction and epoching  
- Labeling epochs based on trigger intervals  
- Converting data into MNE objects  
- ICA and automatic artifact rejection  
- Basic plotting utilities (experimental， now used MNE method)

This package aims to offer a clean and minimal alternative to large EEG frameworks
while remaining fully compatible with **MNE-Python**.

For full function descriptions, see the API documentation (`API.md`).

---

## Installation

### Clone the repository

```bash
git clone https://github.com/songzc18/HEP-EEG.git
cd HEP-EEG
```
---
### Required Python packages:

- numpy
- scipy
- mne
- pyxdf
- PyWavelets
- matplotlib

Install packages: 

```bash
pip install -r requirements.txt
```
---
## Example

Complete usage examples are provided in the `examples/` folder.

- **examples/preprocessing.ipynb**  
  Setup your own pipeline and save epochs.
  
  Complete pipeline: load XDF → preprocess → extract epochs → add labels → save.


- **examples/plotting.ipynb**  
  Select labels and plot HEP/channel-level results.

---

## Project Structure

```
HEP-EEG/
│
├─ data/
│   ├─ parameter/
│   │   ├─ antNeuro63.loc   # channels location
│   │   └─ trigger.csv      # trigger id
│   │ 
│   └─ raw/                 # raw XDF/EEG data
│
├─ examples/
│   ├─ preprocessing.ipynb
│   └─ plotting.ipynb
│
├─ hep_eeg/
│   ├─ hep.py               # main implementation
│   └─ modwt.py             # MODWT implementation
│
├─ README.md
├─ API.md
└─ requirements.txt
```

---

## Acknowledgements

The MODWT implementation used in this toolbox is based on the `modwtpy`
project by GitHub user **pistonly**:

- https://github.com/pistonly/modwtpy

The code was adapted for integration into the HEP-EEG toolbox.

---




