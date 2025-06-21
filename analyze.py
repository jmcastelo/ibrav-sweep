import argparse
import os
import json
import numpy as np
from ase.lattice import CUB, FCC, BCC, HEX, RHL, TET, BCT, ORC, ORCC, ORCF, ORCI, MCL
from yambopy import YamboLatticeDB


def celldm2params(celldm):
    params = {'a': celldm[0]}
    if celldm[1] > 0: params['b'] = celldm[1] * celldm[0]
    if celldm[2] > 0: params['c'] = celldm[2] * celldm[0]
    if celldm[3] != 0: params['gamma'] = np.round(np.arccos(celldm[3]) * 180 / np.pi, 4)
    if celldm[4] != 0: params['beta'] = np.round(np.arccos(celldm[4]) * 180 / np.pi, 4)
    if celldm[5] != 0: params['alpha'] = np.round(np.arccos(celldm[5]) * 180 / np.pi, 4)

    return params



def get_angle_between(ai, aj):
    return np.round(np.arccos(np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj))) * 180 / np.pi, 4)



def get_cell_params(cell):
    cell_params = [
        np.round(np.linalg.norm(cell[:,0]), 4),
        np.round(np.linalg.norm(cell[:,1]), 4),
        np.round(np.linalg.norm(cell[:,2]), 4),
        get_angle_between(cell[:,1], cell[:,2]),
        get_angle_between(cell[:,0], cell[:,2]),
        get_angle_between(cell[:,0], cell[:,1])
    ]
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
        return RHL(params['a'], params['gamma'])
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
        return MCL(params['a'], params['b'], params['c'], params['gamma'])
    elif ibrav == -12:
        return MCL(params['b'], params['a'], params['c'], params['beta'])



def get_rotation(rot_matrix):
    m = rot_matrix - np.transpose(rot_matrix)
    
    rot_vector = np.array([m[2,1], m[0,2], m[1,0]])
    unit_rot_vector = rot_vector / np.linalg.norm(rot_vector)

    rot_angle = np.arccos((np.trace(rot_matrix) - 1) / 2) * 180 / np.pi

    return unit_rot_vector, rot_angle



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Calculation and validation of transformation matrices between ASE and Quantum ESPRESSO crystalline lattice representations')
    parser.add_argument('--input', type = str, required = True, help = 'Input file containing parametric sweep data obtained with Quantum ESPRESSO (JSON format)')
    parser.add_argument('--output', type = str, required = False, help = 'Output file containing resulting matrices and validation tests (JSON format)')

    args = parser.parse_args()

    output_results = []

    p_matrices = []
    r_matrices = []

    input_filename = args.input

    if not os.path.exists(input_filename):
        raise FileNotFoundError('Input file does not exist')

    output_path = args.output

    if not output_path:
        input_path = os.path.split(input_filename)
        output_path = os.path.join(input_path[0], 'r' + input_path[1])

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

        # Get QE equivalent high symmetry points

        database = data['database']

        ylat = YamboLatticeDB.from_db_file(filename = database, Expand = False)

        eq_k_frac_qe = {}

        for i, label in enumerate(k_points):
            eq_k_frac_qe[label] = [np.matmul(k_frac_qe[i], symmetry).tolist() for symmetry in ylat.sym_rec_red]
            #eq_k_frac_qe[label] = [np.matmul(symmetry, k_frac_qe[i]) for symmetry in ylat.sym_rec_red]

        # for i, symmetry in enumerate(ylat.sym_rec_red):
        #     print(i, symmetry)

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

        # Workflow tests

        # Test if direct lattice bases are equal
        test_dir_bases = np.allclose(dir_basis_qe, dir_basis_ase)

        # Test if QE's fractional coordinates of any equivalent high symmetry point
        # coincides with those of its corresponding ASE's k-point
        test_frac_coords = True
        for i, label in enumerate(k_points):
            same_eq_kpt = False
            for eq_kpt in eq_k_frac_qe[label]:
                same_eq_kpt = same_eq_kpt or np.allclose(eq_kpt, k_frac_ase[i], rtol=0, atol=1e-5)
                #print(eq_kpt, k_frac_ase[i])
            test_frac_coords = test_frac_coords and same_eq_kpt

        # Assuming different direct lattice bases, P!=I
        # Assuming same orientation for generated lattices, R=I
        # Obtain P matrix and test if it correctly transforms ASE's fractional coordinates of all high symmetry points
        # to those of the same or equivalent k-points in QE
        p_tmp = np.matmul(np.linalg.inv(dir_basis_ase), dir_basis_qe)
        test_p_matrix = True
        # for i, label in enumerate(k_points):
            # good_p_matrix = False
            # for eq_kpt in eq_k_frac_qe[label]:
            #     good_p_matrix = good_p_matrix or np.allclose(eq_kpt, np.matmul(k_frac_ase[i], p_tmp), rtol=0, atol=1e-5)
            #     print(eq_kpt, np.matmul(k_frac_ase[i], p_tmp))
            # test_p_matrix = test_p_matrix and good_p_matrix
        for i, kpt in enumerate(k_frac_qe):
            test_p_matrix = test_p_matrix and np.allclose(kpt, np.matmul(k_frac_ase[i], p_tmp), rtol=0, atol=1e-5)

        # Assuming same direct lattice bases, P=I
        # Assuming different orientation for generated lattices, R!=I
        # Obtain R matrix and its inverse S, and test if it is orthogonal, and different from the identity
        # and if S correctly transforms ASE's cartesian coordinates of all high symmetry points
        # to those of the same or equivalent k-points in QE
        s_tmp = np.matmul(dir_basis_ase, np.linalg.inv(dir_basis_qe))
        test_r_matrix = True
        # for i, label in enumerate(k_points):
        #     good_r_matrix = False
        #     for eq_kpt in eq_k_frac_qe[label]:
        #         good_r_matrix = good_r_matrix or np.allclose(np.matmul(eq_kpt, rec_basis_qe), np.matmul(k_car_ase[i], s_tmp), rtol = 0, atol = 1e-5)
        #     test_r_matrix = test_r_matrix and good_r_matrix
        for i, kpt in enumerate(k_frac_qe):
            test_r_matrix = test_r_matrix and np.allclose(np.matmul(kpt, rec_basis_qe), np.matmul(k_car_ase[i], s_tmp), rtol = 0, atol = 1e-5)
        r_tmp = np.matmul(dir_basis_qe, np.linalg.inv(dir_basis_ase))

        test_r_ortho = np.allclose(np.transpose(r_tmp), np.linalg.inv(r_tmp), rtol = 0, atol = 1e-5)
        test_r_identity = np.allclose(r_tmp, np.eye(3), rtol = 0, atol = 1e-5)

        test_r_matrix = test_r_matrix and test_r_ortho and not test_r_identity

        # Cases

        cases = [
            '0: A[QE]=A[ASE]',
            '1: F[ASE]=Z.F[QE], P!=I, R=I',
            '2: F[ASE]=Z.F[QE], P=I, R!=I',
            '3: F[ASE]=Z.F[QE], P!=I, R!=I',
            '4: F[ASE]!=Z.F[QE], P!=I, R=I',
            '5: F[ASE]!=Z.F[QE], P!=I, R!=I'
        ]
        case = 0

        p_matrix = np.eye(3)
        r_matrix = np.eye(3)

        rot_vector = np.zeros(3)
        rot_angle = 0

        if test_dir_bases:
            case = 0
            p_matrix = np.eye(3)
            r_matrix = np.eye(3)
            rot_vector = np.zeros(3)
            rot_angle = 0

        elif test_frac_coords:
            if test_p_matrix:
                case = 1
                p_matrix = np.einsum('ij,jk', np.linalg.inv(dir_basis_ase), dir_basis_qe)
                r_matrix = np.eye(3)
                rot_vector = np.zeros(3)
                rot_angle = 0
            elif test_r_matrix:
                case = 2
                p_matrix = np.eye(3)
                r_matrix = np.einsum('ij,jk', dir_basis_qe, np.linalg.inv(dir_basis_ase))
                rot_vector, rot_angle = get_rotation(r_matrix)
            else:
                case = 3
                fb = np.einsum('ki,kj', k_frac_ase, k_frac_ase)
                fb_inv = np.linalg.inv(fb)
                fa = np.einsum('ki,kj', k_frac_ase, k_frac_qe)
                p_matrix = np.einsum('ik,kj', fb_inv, fa)
                r_matrix = np.einsum('ij,jk,kl', dir_basis_qe, np.linalg.inv(p_matrix), np.linalg.inv(dir_basis_ase))
                rot_vector, rot_angle = get_rotation(r_matrix)

        elif test_p_matrix:
            case = 4
            p_matrix = np.einsum('ij,jk', np.linalg.inv(dir_basis_ase), dir_basis_qe)
            r_matrix = np.eye(3)
            rot_vector = np.zeros(3)
            rot_angle = 0

        else:
            case = 5
            fb = np.einsum('ki,kj', k_frac_ase, k_frac_ase)
            fb_inv = np.linalg.inv(fb)
            fa = np.einsum('ki,kj', k_frac_ase, k_frac_qe)
            p_matrix = np.einsum('ik,kj', fb_inv, fa)
            r_matrix = np.einsum('ij,jk,kl', dir_basis_qe, np.linalg.inv(p_matrix), np.linalg.inv(dir_basis_ase))
            rot_vector, rot_angle = get_rotation(r_matrix)

        p_matrices.append(p_matrix)
        r_matrices.append(r_matrix)

        # Validation tests

        test_variant = (variant_qe == variant_ase)

        test_dir_basis = np.allclose(dir_basis_qe, np.einsum('ij,jk,kl', r_matrix, dir_basis_ase, p_matrix))
        test_rec_basis = np.allclose(rec_basis_qe, np.einsum('ij,jk,kl', np.linalg.inv(p_matrix), rec_basis_ase, np.linalg.inv(r_matrix)))

        test_ortho = np.allclose(np.transpose(r_matrix), np.linalg.inv(r_matrix), rtol = 0, atol = 1e-5)

        test_k_frac = True
        for i, label in enumerate(k_points):
            good_k_frac = False
            for eq_kpt in eq_k_frac_qe[label]:
                good_k_frac = good_k_frac or np.allclose(eq_kpt, np.matmul(k_frac_ase[i], p_matrix), rtol = 0, atol = 1e-5)
            test_k_frac = test_k_frac and good_k_frac

        test_k_car = True
        for i, label in enumerate(k_points):
            good_k_car = False
            for eq_kpt in eq_k_frac_qe[label]:
                good_k_car = good_k_car or np.allclose(np.matmul(eq_kpt, rec_basis_qe), np.matmul(k_car_ase[i], np.linalg.inv(r_matrix)), rtol = 0, atol = 1e-5)
            test_k_frac = test_k_frac and good_k_car

        # Results dict

        results = {}
        results['filename'] = filename
        results['variant QE'] = variant_qe
        results['lattice parameters QE'] = lat_params_qe
        results['cell parameters QE'] = np.round(cell_params_qe, 4).tolist()
        results['variant ASE'] = variant_ase
        results['lattice parameters ASE'] = lat_params_ase 
        results['cell parameters ASE'] = np.round(cell_params_ase, 4).tolist()
        results['case'] = cases[case]
        results['symmetries'] = len(ylat.sym_rec_red)
        results['P'] = np.round(p_matrix, 5).tolist()
        results['R'] = np.round(r_matrix, 5).tolist()
        results['rotation vector'] = np.round(rot_vector, 4).tolist()
        results['rotation angle'] = rot_angle
        results['tests'] = {
            'variant identified ?': test_variant,
            'R orthogonal ?': test_ortho,
            'direct basis: A2 = R.A1.P ?': test_dir_basis,
            'reciprocal basis: B2 = Q.B1.S ?': test_rec_basis,
            'k-point fractional coords: F2 = F1.P ?': test_k_frac,
            'k-point Cartesian coords: K2 = K1.S ?': test_k_car
        }
        results['equivalent k-points'] = eq_k_frac_qe

        output_results.append(results)

    # Test P matrices equality 

    all_p_matrices_equal = True

    test_p = []

    for i, p_i in enumerate(p_matrices):
        for j, p_j in enumerate(p_matrices):
            if i < j:
                test = np.allclose(p_i, p_j, atol = 1e-5)
                test_p.append([i, j, test])
                all_p_matrices_equal = all_p_matrices_equal and test

    # Test R matrices equality

    all_r_matrices_equal = True

    test_r = []

    for i, r_i in enumerate(r_matrices):
        for j, r_j in enumerate(r_matrices):
            if i < j:
                test = np.allclose(r_i, r_j, atol = 1e-5)
                test_r.append([i, j, test])
                all_r_matrices_equal = all_r_matrices_equal and test

    output_data = {}
    output_data['all P equal ?'] = all_p_matrices_equal
    output_data['P equality tests'] = test_p
    output_data['all R equal ?'] = all_r_matrices_equal
    output_data['R equality tests'] = test_r
    output_data['results & tests'] = output_results

    with open(output_path, 'w') as file:
        json.dump(output_data, file, indent = 4)
