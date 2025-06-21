import argparse
import os
import shutil
import re
import json
from math import pi, cos



def get_input(calculation, outdir, prefix, ibrav, celldm, ecutwfc, automatic, mesh, k_points):
    """
    Prepare Quantum ESPRESSO input file.

    :param calculation: 'scf' or 'bands'
    :param outdir: Input, temporary, output files are found in this directory
    :param prefix: Prepended to input/output filenames
    :param ibrav: Bravais-lattice index
    :param celldm: Crystallographic constants
    :param ecutwfc: Kinetic energy cutoff (Ry) for wavefunctions
    :param automatic: Automatically generated uniform grid of k-points
    :param mesh: Grid of k-points (to be used if automatic=True)
    :param k_points: K-point labels (to be used if automatic=False)
    :return: Contents of input file
    """
    celldm_fields = [f"{key} = {value}" for key, value in celldm.items()]

    if automatic:
        k_points_card = f"K_POINTS {{ automatic }}\n{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0"
    else:
        k_points_fields = [f"{label} 1" for label in k_points]
        k_points_card = f"K_POINTS {{ crystal_b }}\n{len(k_points)}\n{'\n'.join(k_points_fields)}"

    input_contents = f"""&control
    calculation = '{calculation}'
    outdir = '{outdir}'
    prefix = '{prefix}'
    verbosity = 'high'
    pseudo_dir = './'
/&end
&system
    ibrav = {ibrav}
    {'\n'.join(celldm_fields)}
    ecutwfc = {ecutwfc}
    nat = 1
    ntyp = 1
    occupations = 'smearing'
    degauss = 0.005
/&end
&electrons
    conv_thr = 1e-08
/&end
&ions
    ion_dynamics = 'bfgs'
/&end
ATOMIC_SPECIES
C  10.811  C_ONCV_PBE_sr.upf
ATOMIC_POSITIONS {{ crystal }}
C  0  0  0
{k_points_card}"""

    return input_contents



if __name__ == '__main__':

    sweep_in_data = [
        {   'variant': 'CUB',
            'ibrav': 1,
            'lattice_params': [{'celldm(1)': pi * i} for i in range(1, 4)],
            'k_points': ['M', 'R', 'X']
        },
        {
            'variant': 'FCC',
            'ibrav': 2,
            'lattice_params': [{'celldm(1)': pi * i} for i in range(1, 4)],
            'k_points': ['K', 'L', 'U', 'W', 'X']
        },
        {
            'variant': 'BCC',
            'ibrav': 3,
            'lattice_params': [{'celldm(1)': pi * i} for i in range(1, 4)],
            'k_points': ['H', 'P', 'N']
        },
        {
            'variant': 'HEX',
            'ibrav': 4,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(3)': 2} for i in range(1, 4)],
            'k_points': ['A', 'H', 'K', 'L', 'M'],
            'mesh': [6, 6, 6]
        },
        {
            'variant': 'RHL1',
            'ibrav': 5,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(4)': cos((i / 4) * pi / 2)} for i in range(1, 4)],
            'k_points': ['B', 'B1', 'F', 'L', 'L1', 'P', 'P1', 'P2', 'Q', 'X', 'Z']
        },
        {
            'variant': 'RHL2',
            'ibrav': 5,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(4)': cos((1 + i / 10) * pi / 2)} for i in range(1, 4)],
            'k_points': ['F', 'L', 'P', 'P1', 'Q', 'Q1', 'Z']
        },
        {
            'variant': 'TET',
            'ibrav': 6,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(3)': 1.5} for i in range(1, 4)],
            'k_points': ['A', 'M', 'R', 'X', 'Z']
        },
        {
            'variant': 'BCT1',
            'ibrav': 7,
            'lattice_params': [{'celldm(1)': pi * i * 3 / 2, 'celldm(3)': 2 / 3} for i in range(1, 4)],
            'k_points': ['M', 'N', 'P', 'X', 'Z', 'Z1']
        },
        {
            'variant': 'BCT2',
            'ibrav': 7,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(3)': 3 / 2} for i in range(1, 4)],
            'k_points': ['N', 'P', 'gS', 'gS1', 'X', 'Y', 'Y1', 'Z']
        },
        {
            'variant': 'ORC',
            'ibrav': 8,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 1.5, 'celldm(3)': 2} for i in range(1, 4)],
            'k_points': ['R', 'S', 'T', 'U', 'X', 'Y', 'Z']
        },
        {
            'variant': 'ORCC',
            'ibrav': 9,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 1.3, 'celldm(3)': 1.7} for i in range(1, 4)],
            'k_points': ['A', 'A1', 'R', 'S', 'T', 'X', 'X1', 'Y', 'Z']
        },
        {
            'variant': 'ORCF1',
            'ibrav': 10,
            'lattice_params': [{'celldm(1)': 0.7 * pi * i, 'celldm(2)': 5 / (4 * 0.7), 'celldm(3)': 5 / (3 * 0.7)} for i in range(1, 4)],
            'k_points': ['A', 'A1', 'L', 'T', 'X', 'X1', 'Y', 'Z']
        },
        {
            'variant': 'ORCF2',
            'ibrav': 10,
            'lattice_params': [{'celldm(1)': 1.2 * pi * i / 2, 'celldm(2)': 5 / (4 * 1.2), 'celldm(3)': 5 / (3 * 1.2)} for i in range(1, 4)],
            'k_points': ['C', 'C1', 'D', 'D1', 'L', 'H', 'H1', 'X', 'Y', 'Z']
        },
        {
            'variant': 'ORCF3',
            'ibrav': 10,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 5 / 4, 'celldm(3)': 5 / 3} for i in range(1, 4)],
            'k_points': ['A', 'A1', 'L', 'T', 'X', 'Y', 'Z'],
            'ecutwfc': 50
        },
        {
            'variant': 'ORCI',
            'ibrav': 11,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 1.3, 'celldm(3)': 1.7} for i in range(1, 4)],
            'k_points': ['L', 'L1', 'L2', 'R', 'S', 'T', 'W', 'X', 'X1', 'Y', 'Y1', 'Z']
        },
        {
            'variant': 'MCL',
            'ibrav': 12,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 1.3, 'celldm(3)': 1.7, 'celldm(4)': cos((i / 4) * pi / 2)} for i in range(1, 4)],
            'k_points': ['A', 'D', 'X', 'Y', 'Z'],
            'mesh': [2, 2, 2]
        },
        {
            'variant': 'MCL',
            'ibrav': -12,
            'lattice_params': [{'celldm(1)': pi * i, 'celldm(2)': 1.3, 'celldm(3)': 1.7, 'celldm(5)': cos((i / 4) * pi / 2)} for i in range(1, 4)],
            'k_points': ['A', 'D', 'X', 'Y', 'Z'],
            'mesh': [2, 2, 2]
        }
    ]

    sweeps_dir = 'sweeps'
    
    parser = argparse.ArgumentParser(description = 'Parametric sweep of Bravais lattice parameters using Quantum ESPRESSO (scf + bands), and data extraction for selected Bravais lattice types')
    parser.add_argument('--sweep', action = argparse.BooleanOptionalAction, required = False, help = 'Perform sweep on lattice parameters of selected Bravais lattice')
    parser.add_argument('--ibrav', type = int, required = False, help = 'Specify Bravais lattice index')
    parser.add_argument('--variant', type = str, required = False, help = 'Specify Bravais lattice variant')
    parser.add_argument('--sweep_output', type = str, required = False, help = 'Specify directory where to save sweep input and output files (Caution: overwrites dir!)')
    parser.add_argument('--parse', action = argparse.BooleanOptionalAction, required = False, help = 'Parse all relevant output data files produced by parametric sweep')
    parser.add_argument('--parse_input', type = str, required = False, help = 'Specify input directory where to find parametric sweep files')
    parser.add_argument('--parse_output', type = str, required = False, help = 'Specify output file where to save extracted data (JSON format)')

    args = parser.parse_args()

    # Check ibrav and variant arguments

    ibrav = args.ibrav
    variant = args.variant

    if variant:
        variant = variant.upper()

    if ibrav is not None:
        ibrav_matches = [entry for entry in sweep_in_data if entry['ibrav'] == ibrav]
        if len(ibrav_matches) == 0:
            raise ValueError('Bravais lattice index not recognized')
        elif len(ibrav_matches) == 1:
            variant = ibrav_matches[0]['variant']
        elif variant is not None:
            variant_matches = [entry for entry in ibrav_matches if entry['variant'] == variant]
            if len(variant_matches) == 0:
                raise ValueError('Bravais lattice variant not recognized')
        else:
            raise ValueError('Must specify Bravais lattice variant')
    elif variant is not None:
        variant_matches = [entry for entry in sweep_in_data if entry['variant'] == variant]
        if len(variant_matches) == 0:
            raise ValueError('Bravais lattice variant not recognized')
        elif len(variant_matches) == 1:
            ibrav = variant_matches[0]['ibrav']
        elif ibrav is not None:
            ibrav_matches = [entry for entry in sweep_in_data if entry['ibrav'] == ibrav]
            if len(ibrav_matches) == 0:
                raise ValueError('Bravais lattice index not recognized')
        else:
            raise ValueError('Must specify Bravais lattice index')



    # Parametric sweep

    if args.sweep:
        match = [entry for entry in sweep_in_data if entry['ibrav'] == ibrav and entry['variant'] == variant]

        input_data = match[0]

        k_points = input_data['k_points']
    
        outdir = args.sweep_output

        if not outdir:
            outdir = f"{sweeps_dir}/{variant}_{ibrav}"
    
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        
        os.makedirs(outdir)

        # Save variant

        with open(f"{outdir}/variant", 'w') as file:
            file.write(f"{variant}")
     
        # Loop over lattice parameters

        for index, celldm in enumerate(input_data['lattice_params']):
            i = index + 1
            
            print(f"[{i}] (ibrav: {ibrav}, variant: {variant}) {celldm}")

            prefix = f"{variant}_{i:02d}"
            
            ecutwfc = input_data.get('ecutwfc', 80)
            mesh = input_data.get('mesh', [1, 1, 1])
    
            # SCF

            scf_input = get_input('scf', outdir, prefix, ibrav, celldm, ecutwfc, True, mesh, k_points)
    
            scf_filename = f"{outdir}/scf_{variant}_{i:02d}"
        
            with open(f"{scf_filename}.in", 'w') as file:
                file.write(scf_input)
    
            print('sfc')

            status = os.system(f"pw.x < {scf_filename}.in > {scf_filename}.out")

            if status != 0:
                print('\terror')
                break
    
            print('\tdone')

            # p2y

            print('p2y')

            save_dir = f"{outdir}/{prefix}.save"

            if not os.path.exists(save_dir):
                raise FileNotFoundError('save dir not found')

            status = os.system(f"p2y -nompi -I {save_dir} -O {save_dir}")

            if status != 0:
                print('\terror')
            else:
                print('\tdone')

            # Bands

            bands_input = get_input('bands', outdir, prefix, ibrav, celldm, ecutwfc, False, mesh, k_points)
    
            bands_filename = f"{outdir}/bands_{variant}_{i:02d}"
    
            with open(f"{bands_filename}.in", 'w') as file:
                file.write(bands_input)
    
            print('bands')

            status = os.system(f"pw.x < {bands_filename}.in > {bands_filename}.out")

            if status != 0:
                print('\terror')
            else:
                print('\tdone')
    
    # Parse band calculation output files
    
    if args.parse:
        indir = args.parse_input

        if not indir:
            indir = f"{sweeps_dir}/{variant}_{ibrav}"

        if not os.path.exists(indir):
            raise FileNotFoundError('Input dir does not exist')

        # Get variant

        variant_filepath = os.path.join(indir, 'variant')

        if not os.path.exists(variant_filepath):
            raise FileNotFoundError('Variant file does not exist')

        variant = ''

        with open(variant_filepath, 'r') as file:
            variant = file.readline().strip('\n')

        # Get band output filenames and sort them

        filepaths = []

        for entry in os.scandir(indir):
            if entry.is_file() and entry.name.startswith('bands') and entry.name.endswith('.out'):
                filepaths.append(os.path.join(indir, entry.name))
        
        if len(filepaths) == 0:
            print('No Quantum ESPRESSO band calculation output files found')
            exit()

        filepaths = sorted(filepaths)

        databases = []

        for path, dirs, files in os.walk(indir):
            if 'ns.db1' in files:
                databases.append(os.path.join(path, 'ns.db1'))

        databases = sorted(databases)

        print(f"Parsing {len(filepaths)} output files:")

        sweep_out_data = []

        for i, filepath in enumerate(filepaths):

            parsed_data = {}

            print(filepath)

            ibrav = 0
            celldm = []
            a_vectors = []
            b_vectors = []
            k_points = []

            ibrav_pattern = re.compile(r'bravais-lattice index\s*=\s*(.*)')
            celldm_pattern = re.compile(r'celldm\(\d+\)=\s*(-?\d+\.\d+)')
            a_pattern = re.compile(r'a\(\d+\) = \((.*)\)')
            b_pattern = re.compile(r'b\(\d+\) = \((.*)\)')
            k_pattern = re.compile(r'k\((.*)\) = \((.*)\)')
            
            with open(filepath, 'r') as file:
                for line in file:
                    ibrav_result = ibrav_pattern.search(line)
                    celldm_result = celldm_pattern.findall(line)
                    a_result = a_pattern.search(line)
                    b_result = b_pattern.search(line)
                    k_result = k_pattern.search(line)

                    if ibrav_result:
                        ibrav = int(ibrav_result.group(1))
                    if celldm_result:
                        celldm.append([float(c) for c in celldm_result])
                    if a_result:
                        a_vectors.append([float(a) for a in a_result.group(1).split()])
                    if b_result:
                        b_vectors.append([float(b) for b in b_result.group(1).split()])
                    if k_result:
                        k_points.append([float(k) for k in k_result.group(2).split()])

            
            parsed_data['filename'] = filepath
            parsed_data['database'] = databases[i]
            parsed_data['variant'] = variant
            parsed_data['ibrav'] = ibrav
            parsed_data['celldm'] = [c for sublist in celldm for c in sublist]
            parsed_data['direct_basis'] = a_vectors
            parsed_data['reciprocal_basis'] = b_vectors

            for entry in sweep_in_data:
                if entry['variant'] == variant and entry['ibrav'] == ibrav:
                    parsed_data['k_points'] = entry['k_points']

            half = len(k_points) // 2
            parsed_data['k_coords_cartesian'] = k_points[:half]
            parsed_data['k_coords_fractional'] = k_points[half:]

            sweep_out_data.append(parsed_data)
        
        parse_output = args.parse_output

        if not parse_output:
            parse_output = f"{sweeps_dir}/{variant}_{ibrav}.json"

        with open(parse_output, 'w') as file:
            json.dump(sweep_out_data, file, indent = 4)


