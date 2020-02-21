import csv
import os

import numpy as np

from morphic import morphic as morphic


LEFT_MUSCLE_ELEMENTS = [
    ['7', '5', '19', '17', '8', '6', '20', '18'],
    ['9', '7', '21', '19', '10', '8', '22', '20'],
    ['11', '9', '23', '21', '12', '10', '24', '22'],
    ['19', '17', '31', '29', '20', '18', '32', '30'],
    ['21', '19', '33', '31', '22', '20', '34', '32'],
    ['23', '21', '35', '33', '24', '22', '36', '34'],
    ['27', '25', '39', '37', '28', '26', '40', '38'],
    ['29', '27', '41', '39', '30', '28', '42', '40'],
    ['31', '29', '43', '41', '32', '30', '44', '42'],
    ['33', '31', '45', '43', '34', '32', '46', '44'],
    ['35', '33', '47', '45', '36', '34', '48', '46'],
    ['41', '39', '53', '51', '42', '40', '54', '52'],
    ['43', '41', '55', '53', '44', '42', '56', '54'],
    ['45', '43', '57', '55', '46', '44', '58', '56'],
    ['47', '45', '59', '57', '48', '46', '60', '58'],
    ['55', '53', '67', '65', '56', '54', '68', '66'],
    ['57', '55', '69', '67', '58', '56', '70', '68'],
    ['59', '57', '71', '69', '60', '58', '72', '70'],
    ['69', '67', '81', '79', '70', '68', '82', '80'],
    ['71', '69', '83', '81', '72', '70', '84', '82']
]

SUBJECTS = ['01', '08', '14', '17', '20', '31', '34', '36', '40']


class MESH(object):

    def __init__(self, file_path):
        self._muscle = None
        self.count = 0
        self._elements = LEFT_MUSCLE_ELEMENTS
        self._mesh = None
        self.output = None
        self.file_path = file_path
        self.mesh = None

    def generate_mesh(self, muscle='L', save=True):
        if muscle == 'L' or muscle == 'l':
            self._muscle = 'Left'
        elif muscle == 'R' or muscle == 'r':
            self._muscle = 'Right'
        elif muscle == 'LR' or muscle == 'lr' or muscle == 'RL' or muscle == 'rl':
            self._muscle = 'Muscle'

        if self._mesh is not None:
            self._mesh = None

        self._mesh = morphic.Mesh()
        data = {}

        print('\n\t=========================================\n')
        print('\t   GENERATING MESH... \n')
        print('\t   PLEASE WAIT... \n')

        for filenum in os.listdir(self.file_path):
            if filenum.split('_')[1] in SUBJECTS:
                mesh_path = os.path.join(self.file_path, filenum, 'FEMesh', self._muscle + '_fitted.node')
                if os.path.isfile(mesh_path):
                    self.count += 1
                    with open(mesh_path, 'r') as csvfile:
                        data[filenum] = csv.reader(csvfile, delimiter=' ', quotechar='|')
                        for rowx in data[filenum]:
                            rowy = next(data[filenum])
                            rowz = next(data[filenum])
                            node = [[float(rowx[1]), float(rowx[2]), float(rowx[3]), float(rowx[4])],
                                    [float(rowy[1]), float(rowy[2]), float(rowy[3]), float(rowy[4])],
                                    [float(rowz[1]), float(rowz[2]), float(rowz[3]), float(rowz[4])]]
                            nd = self._mesh.add_stdnode(str(rowx[0]), node)

                        if self._muscle == 'Left':
                            elements = self._elements
                        else:
                            raise ValueError("Only left muscle elements are valid.")

                        for ii, elem in enumerate(elements):
                            self._mesh.add_element(ii + 1, ['H3', 'H3', 'H3'], elem)

                        self._mesh.generate()

                        if save:
                            mesh_output = os.path.normpath(mesh_path + os.sep + os.pardir)
                            self._mesh.save(mesh_output + '/' + self._muscle + '_fitted.mesh', format='h5py')

                            print('\t   MESH SAVED IN \n')
                            print('\t   {} DIRECTORY \n'.format(str(mesh_output)))
        print('\n\t=========================================\n')


def align_mesh(reference_mesh, mesh, sub, scaling=False, reflection='best'):

    print('\n\t=========================================\n')
    print('\t   ALIGNING MESH... \n')
    print('\t   PLEASE WAIT... \n')

    r = morphic.Mesh(reference_mesh)
    mesh = morphic.Mesh(mesh)

    X = r.get_nodes()
    Y = mesh.get_nodes()

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    """ centred Frobenius norm """
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    """ scale to equal (unit) norm """
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    """ optimum rotation matrix of Y """
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        """ if the current solution use a reflection? """
        have_reflection = np.linalg.det(T) < 0

        """ if that's not what was specified, force another reflection """
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        """ optimum scaling of Y """
        b = traceTA * normX / normY

        """ standarised distance between X and b*Y*T + c """
        d = 1 - traceTA ** 2

        """ transformed coords """
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    """ translation matrix """
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    """ transformation values """
    tform = {'rotation': T, 'scale': b, 'translation': c}

    node_list = []
    for num, object in enumerate(mesh.nodes):
        node = mesh.nodes[object.id].values[:, 0]
        Zlist = Z.tolist()
        mesh.nodes[object.id].values[:, 0] = Zlist[num]
        node_list.append(object.id)

    output = 'morphic_aligned'

    mesh_output = os.path.join('Z:\\pectoral_muscle\\fitting\\cubic_hermite\\data', sub, 'FEMesh', output)

    if not os.path.exists(mesh_output):
        os.makedirs(mesh_output)

    mesh.save(os.path.join(mesh_output, 'aligned_Left_fitted.mesh'), format='h5py')

    print('\t   ALIGNED MESH SAVED IN \n')

    print('\n\t=========================================\n')

    return d, Z, tform, mesh, node_list


if __name__ == '__main__':
    path = 'Z:\\pectoral_muscle\\fitting\\cubic_hermite\\data\\'
    m = MESH(path)
    m.generate_mesh()

    reference = os.path.join(path, 'subject_08', 'FEMesh', 'Left_fitted.mesh')

    for filenum in os.listdir(path):
        if filenum.split('_')[1] in SUBJECTS:
            mesh_path = os.path.join(path, filenum, 'FEMesh', 'Left_fitted.mesh')
            print(filenum)
            if os.path.isfile(mesh_path):
                d, Z, tform, mesh, node_list = align_mesh(reference, mesh_path, filenum)
