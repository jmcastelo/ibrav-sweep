import argparse
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from ase.lattice import CUB, FCC, BCC, HEX, RHL, TET, BCT, ORC, ORCC, ORCF, ORCI, MCL
from yambopy import YamboLatticeDB


def celldm2params(celldm):
    params = {'a': celldm[0]}
    if celldm[1] > 0: params['b'] = celldm[1] * celldm[0]
    if celldm[2] > 0: params['c'] = celldm[2] * celldm[0]
    if celldm[3] != 0: params['alpha'] = np.round(np.arccos(celldm[3]) * 180 / np.pi, 4)
    if celldm[4] != 0: params['beta'] = np.round(np.arccos(celldm[4]) * 180 / np.pi, 4)
    if celldm[5] != 0: params['gamma'] = np.round(np.arccos(celldm[5]) * 180 / np.pi, 4)

    return params



def get_angle_between(ai, aj):
    return np.round(np.arccos(np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj))) * 180 / np.pi, 4)



def get_cell_params(cell):
    cell_params = (
        np.round(np.linalg.norm(cell[:,0]), 4),
        np.round(np.linalg.norm(cell[:,1]), 4),
        np.round(np.linalg.norm(cell[:,2]), 4),
        get_angle_between(cell[:,1], cell[:,2]),
        get_angle_between(cell[:,0], cell[:,2]),
        get_angle_between(cell[:,0], cell[:,1])
    )
    return cell_params
    


def get_bravais_lattice(ibrav, params):
    if ibrav == 1:
        return CUB(params['a'])
    elif ibrav == 2:
        return FCC(params['a'])
    elif ibrav == 3:
        return BCC(params['a'])
    elif ibrav == 4:
        return HEX(params['a'], params['c'])
    elif ibrav == 5:
        return RHL(params['a'], params['alpha'])
    elif ibrav == 6:
        return TET(params['a'], params['c'])
    elif ibrav == 7:
        return BCT(params['a'], params['c'])
    elif ibrav == 8:
        return ORC(params['a'], params['b'], params['c'])
    elif ibrav == 9:
        return ORCC(params['a'], params['b'], params['c'])
    elif ibrav == 10:
        return ORCF(params['a'], params['b'], params['c'])
    elif ibrav == 11:
        return ORCI(params['a'], params['b'], params['c'])
    elif ibrav == 12:
        return MCL(params['a'], params['b'], params['c'], params['alpha'])
    elif ibrav == -12:
        return MCL(params['a'], params['b'], params['c'], params['beta'])



def get_rotation(rot_matrix):
    m = rot_matrix - np.transpose(rot_matrix)
    
    rot_vector = np.array([m[2,1], m[0,2], m[1,0]])
    unit_rot_vector = rot_vector / np.linalg.norm(rot_vector)

    rot_angle = np.arccos((np.trace(rot_matrix) - 1) / 2) * 180 / np.pi

    return unit_rot_vector, rot_angle



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Plot high symmetry points')
    parser.add_argument('--input', type = str, required = True, help = 'Input file containing parametric sweep data obtained with Quantum ESPRESSO (JSON format)')

    args = parser.parse_args()

    input_filename = args.input

    if not os.path.exists(input_filename):
        raise FileNotFoundError('Input file does not exist')

    sweep_out_data = []

    with open(input_filename, 'r') as file:
        sweep_out_data = json.load(file)

    for data in sweep_out_data:
        
        # Metadata

        filename = data['filename']

        # Quantum ESPRESSO

        variant_qe = data['variant']
        ibrav = data['ibrav']

        celldm_qe = data['celldm']
        alat_qe = celldm_qe[0]
        
        dir_basis_qe = np.transpose(np.array(data['direct_basis'])) * alat_qe
        rec_basis_qe = np.array(data['reciprocal_basis']) / alat_qe

        lat_params_qe = celldm2params(celldm_qe)
        cell_params_qe = get_cell_params(dir_basis_qe)
        
        k_points = [k.lstrip('g') for k in data['k_points']]
        k_frac_qe = np.array(data['k_coords_fractional'])
        k_car_qe = np.array(data['k_coords_cartesian']) / alat_qe

        # ASE/SC

        blat = get_bravais_lattice(ibrav, lat_params_qe)
        
        variant_ase = blat.variant

        cell = blat.tocell()

        lat_params_ase = blat.vars()
        cell_params_ase = cell.cellpar()

        dir_basis_ase = np.transpose(cell[:])
        rec_basis_ase = cell.reciprocal()[:]

        k_points_ase = blat.get_special_points()
        k_frac_ase = np.array([k_points_ase[k] for k in k_points])

        path = ''.join(k_points)
        bandpath = blat.bandpath(path = path, npoints = 0)
        k_car_ase = bandpath.cartesian_kpts()

        # Get QE equivalent high symmetry points

        database = data['database']

        ylat = YamboLatticeDB.from_db_file(filename = database, Expand = False)

        eq_k_frac_qe = {}
        eq_k_car_qe = {}

        for i, label in enumerate(k_points):
            eq_k_frac_qe[label] = [np.matmul(symmetry, k_frac_qe[i]) for symmetry in ylat.sym_rec_red]
            eq_k_car_qe[label] = np.einsum('ij,jk', eq_k_frac_qe[label], rec_basis_qe)

        for i, symmetry in enumerate(ylat.sym_rec_red):
            print(i+1, symmetry)

        # Plot reciprocal bases and high symmetry points

        k_ase = np.einsum('ij,jk', k_frac_ase, rec_basis_ase)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_box_aspect((1,1,1))

        for i in range(3):
            ax.text(rec_basis_qe[i,0], rec_basis_qe[i,1], rec_basis_qe[i,2], i+1, color='red')
            ax.plot([0, rec_basis_qe[i,0]], [0, rec_basis_qe[i,1]], [0, rec_basis_qe[i,2]], color='red')
            ax.text(rec_basis_ase[i,0], rec_basis_ase[i,1], rec_basis_ase[i,2], i+1, color='blue')
            ax.plot([0, rec_basis_ase[i,0]], [0, rec_basis_ase[i,1]], [0, rec_basis_ase[i,2]], color='blue')

        ax.scatter(k_ase[:, 0], k_ase[:, 1], k_ase[:, 2], marker='^', color='blue')

        for label, kpts_qe in eq_k_car_qe.items():
            ax.scatter(kpts_qe[:,0], kpts_qe[:,1], kpts_qe[:,2], marker='o', color='red')
            for kpt in kpts_qe:
                ax.text(kpt[0], kpt[1], kpt[2], label)

        for i, label in enumerate(k_points):
            ax.text(k_ase[i,0], k_ase[i,1], k_ase[i,2], label)

        ax.set_xlabel('Kx')
        ax.set_ylabel('Ky')
        ax.set_zlabel('Kz')

        plt.show()
