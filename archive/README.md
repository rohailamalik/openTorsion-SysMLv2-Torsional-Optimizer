# IFTOMM Project Setup

## Setup Instructions

### 1. Clone the repository

```bash
mkdir ~/iftomm
cd ~/iftomm
git clone https://github.com/rohailamalik/iftomm.git .
```

### 2. Clone openTorsion dependency

```bash
git clone https://github.com/Aalto-Arotor/openTorsion.git
echo "openTorsion/" >> .gitignore
```

### 3. Install openTorsion

```bash
pip install ./openTorsion
```

### 4. Run the project

```bash
python parse2.py
```

The assembler.py assembles a system, the placeHolder.json has a engine-coupling-propeller with the coupling being a placeholde. 

The centa_coupling.json has couplings from SysMLv2 API, the examples here are size 16, page 15 of the CENTA document.

The analysis_cases/ folder contains the full system with each coupling inserted. This is the system the assembler makes.


## Troubleshooting

If imports fail, verify installation:
```bash
pip list | grep opentorsion
```

To update openTorsion:
```bash
cd openTorsion
git pull
cd ..
pip install ./openTorsion
```
