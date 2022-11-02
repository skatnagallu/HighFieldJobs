# coding: utf-8
# Copyright (c) - Max-Planck-Institut für Eisenforschung GmbH Computational Materials Design (CM) Department, MPIE.
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import subprocess
from pyiron_atomistics.sphinx.base import Group
from pyiron_atomistics.atomistics.structure.atoms import pymatgen_to_pyiron, pyiron_to_pymatgen, ase_to_pyiron, \
    CrystalStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.build import surface, bulk, add_adsorbate

__author__ = "Shyam Katnagallu"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.9"
__maintainer__ = "Shyam Katnagallu"
__email__ = "s.katnagallu@mpie.de"
__status__ = "production"
__date__ = "Nov 3, 2021"


class HighFieldJob:
    num_of_jobs = 0
    HARTREE_TO_EV = 27.2114
    ANGSTROM_TO_BOHR = 1.8897
    preconditioner = 'ELLIPTIC'
    rhomixing = str(0.7)
    preconscaling = 0.3
    threads = 4
    cores = 20
    ekt = 0.1
    ekt_scheme = 'Fermi'
    e_energy = 1e-7
    i_energy = 1e-3

    def __init__(self, pr, e_field, en_cut, k_cut):
        """ HighFieldJob instance which has pr a pyiron project attribute, structure attribute, job_name attribute,
            eField as the electric field (V/A) to be applied attribute, en_cut as the energy cutoff attribute in eV and
            k_cut as the k_point mesh. """
        self.pr = pr
        self.e_field = e_field
        self.en_cut = en_cut
        self.k_cut = k_cut
        HighFieldJob.num_of_jobs += 1

    def reset(self, pr, e_field, en_cut, k_cut):
        """ reset high field job attributes."""
        self.pr = pr
        self.e_field = e_field
        self.en_cut = en_cut
        self.k_cut = k_cut

    @classmethod
    def set_job_variables(cls, rho_mixing, preconditioner, preconscaling, threads, cores, ekt, ekt_scheme):
        cls.preconditioner = preconditioner
        cls.preconscaling = preconscaling
        cls.rho_mixing = rho_mixing
        cls.threads = threads
        cls.cores = cores
        cls.ekt = ekt
        cls.ekt_scheme = ekt_scheme

    def slab_height_jobs(self, structure, job_name, number_layers):
        """ Function to create jobs and run for slab height convergence
        :param structure: primitive structure, which will be repeated [1,1,number_layers]  for calculations
        :param job_name: Job_name describing the structure and e_field for instance
        :param number_layers: a list of integers indicating number of layers in z
        :return: None runs jobs
        """
        self.pr = self.pr.create_group("slab_heights")
        for i in number_layers:
            size = [1, 1, int(i)]
            struct = structure.repeat(size)
            job = self.gdc_relaxation(structure=struct, job_name=job_name+'_'+str(int(i))+'layers', z_height=2)
            job.run()

    def frozen_height_jobs(self, structure, job_name, frozen_height):
        """ Function to create jobs and run for frozen height convergence
        :param structure: primitive structure, which will be repeated [1,1,number_layers]  for calculations
        :param job_name: Job_name describing the structure and e_field for instance
        :param frozen_height: a list of heights, to be kept fixed in slab
        :return: None runs jobs
        """
        self.pr = self.pr.create_group("frozen_heights")
        for i in frozen_height:
            struct = structure.copy()
            job = self.gdc_relaxation(structure=struct, job_name=job_name+'_'+str(int(i))+'fh', z_height=i)
            job.run()

    @staticmethod
    def frozen_height_analysis(pr):
        """
        A function to analyse the frozen height convergence jobs for gdc.
        :param pr: Pyiron project folder where the jobs are contained
        :return: relaxPositions and relaxPatterns with changes in positions and in positions in percentage.
        """
        relaxPatterns = {}
        relaxPositions = {}
        for job in pr.iter_jobs():
            if job.status == 'finished':
                ini = np.diff(job.get_structure(0).positions[:, 2])
                fin = np.diff(job.get_structure(-1).positions[:, 2])
                relaxPatterns[job.job_name.split('_', -1)[-1]] = ((fin - ini) / ini) * 100
                relaxPositions[job.job_name.split('_', -1)[-1]] = (fin - ini)

        return relaxPositions, relaxPatterns

    @staticmethod
    def slab_height_analysis(pr):
        """
        A function to analyse the slab height convergence jobs for gdc.
        :param pr: Pyiron project folder where the jobs are contained
        :return: relaxPositions and relaxPatterns with changes in positions and in positions in percentage.
        """
        relaxPatterns = {}
        relaxPositions = {}
        for job in pr.iter_jobs():
            if job.status == 'finished':
                ini = np.diff(job.get_structure(0).positions[:, 2])
                fin = np.diff(job.get_structure(-1).positions[:, 2])
                relaxPatterns[job.job_name.split('_', -1)[-1]] = ((fin - ini) / ini) * 100
                relaxPositions[job.job_name.split('_', -1)[-1]] = (fin - ini)

        return relaxPositions, relaxPatterns

    def gdc_evaporation(self, structure, job_name, index, z_height=2, vdw=False, PES_xy=False, TS=False):
        """Function to set up charged slab calculations with eField in Volts/Angstrom, and fixing layers below the
        specified z_height (Angstroms). The function take HighFieldJob instance as input with additional arguments
        of index for field evaporating atom. Set PES_xy to True to calculate PES along xy.

        :param TS: Set to True for transition state optimization calculation
        :param PES_xy: Set to True for potential energy surface calculation along xy
        :param vdw: set to True to include van derWalls corrections of D2 type
        :param z_height: height of layers to be fixed
        :param index: index of the evaporating (interested) atom
        :param job_name: name of pyiron job
        :param structure: pyiron structure object

        :return: a pyiron job
        """
        job = self.pr.create_job(
            job_type=self.pr.job_type.Sphinx,
            job_name=job_name
        )
        job.set_occupancy_smearing(self.ekt_scheme, width=self.ekt)
        job.structure = structure
        job.set_encut(self.en_cut)  # in eV
        job.set_kpoints(self.k_cut, center_shift=[0.5, 0.5, 0.25])
        job.set_convergence_precision(electronic_energy=self.e_energy, ionic_energy_tolerance=self.i_energy)
        positions = [p[2] for p in job.structure.positions]
        job.structure.add_tag(selective_dynamics=(True, True, True))
        job.structure.selective_dynamics[
            np.where(np.asarray(positions) < z_height)[0]
        ] = (False, False, False)
        if PES_xy:
            job.structure.selective_dynamics[index] = (False, False, True)
        job.structure.selective_dynamics[index] = (True, True, False)
        job.calc_minimize(ionic_steps=100,
                          electronic_steps=100)
        right_field = self.e_field / 51.4  # atomic units (1 E_h/ea_0 ~= 51.4 V/Å)
        left_field = 0.0
        cell = job.structure.cell * self.ANGSTROM_TO_BOHR
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        total_charge = (right_field - left_field) * area / (4 * np.pi)
        sort_positions = np.sort(positions)
        job.input.sphinx.initialGuess.rho.charged = Group({})
        job.input.sphinx.initialGuess.rho.charged.charge = total_charge
        job.input.sphinx.initialGuess.rho.charged.z = sort_positions[-2] * self.ANGSTROM_TO_BOHR
        if vdw:
            job.input.sphinx.PAWHamiltonian.vdwCorrection = Group({})
            job.input.sphinx.PAWHamiltonian.vdwCorrection.method = "\"D2\""
        job.input.sphinx.PAWHamiltonian.nExcessElectrons = -total_charge
        job.input.sphinx.PAWHamiltonian.dipoleCorrection = True
        job.input.sphinx.PAWHamiltonian.zField = left_field * self.HARTREE_TO_EV
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag'][
            'rhoMixing'] = self.rhomixing  # using conservative mixing can help with convergence.
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'type'] = self.preconditioner
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'scaling'] = self.preconscaling
        job.input['THREADS'] = self.threads
        if TS:
            mainGroup = job.input.sphinx['main']
            # rename ricQN group inside mainGroup to ricTS group
            mainGroup['ricTS'] = mainGroup.pop('ricQN')
            # now add the transition path group inside ricTS
            # atomId must be the atom index of the evaporating atom
            mainGroup['ricTS'].set_group('transPath')
            tp = mainGroup['ricTS']['transPath']
            tp['atomId'] = index + 1  # python to sphinx change in indices
            tp['dir'] = [0, 0, 1]
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag'][
                'rhoMixing'] = self.rhomixing  # using conservative mixing can help with convergence.
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag']['preconditioner'][
                'type'] = self.preconditioner
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag']['preconditioner'][
                'scaling'] = self.preconscaling
        queue = 'cm'
        job.server.cores = self.cores
        job.server.queue = queue
        return job

    def field_free_relaxation(self, structure, job_name, z_height=2, vdw=False, constrained=False, index=None):
        """Function to set up  slab relaxation calculations for the given HighFieldJob instance, by fixing the
        layers lying lower than z_height (Angstroms) and without any field. Use constrained if you want to fix an atom
        of index."""
        job = self.pr.create_job(self.pr.job_type.Sphinx, job_name)
        job.set_occupancy_smearing(self.ekt_scheme, width=self.ekt)
        job.structure = structure
        positions = [p[2] for p in job.structure.positions]
        job.structure.add_tag(selective_dynamics=(True, True, True))
        job.structure.selective_dynamics[
            np.where(np.asarray(positions) < z_height)[0]
        ] = (False, False, False)
        if constrained:
            job.structure.selective_dynamics[index] = (False, False, True)
        job.set_kpoints(self.k_cut)
        job.set_encut(self.en_cut)
        job.set_convergence_precision(electronic_energy=self.e_energy, ionic_energy_tolerance=self.i_energy)
        job.calc_minimize()
        if vdw:
            job.input.sphinx.PAWHamiltonian.vdwCorrection = Group({})
            job.input.sphinx.PAWHamiltonian.vdwCorrection.method = "\"D2\""
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag'][
            'rhoMixing'] = self.rhomixing  # using conservative mixing can help with convergence.
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'type'] = self.preconditioner
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'scaling'] = self.preconscaling
        job.fix_symmetry = False
        job.input['THREADS'] = self.threads
        queue = 'cm'
        job.server.cores = self.cores
        job.server.queue = queue
        return job

    def gdc_relaxation(self, structure, job_name, z_height=2, vdw=False, constrained=False, index=None):
        """Function to set up charged slab relaxation calculations for the given HighFieldJob instance, by fixing the
        layers lying lower than z_height (Angstroms)."""
        job = self.pr.create_job(self.pr.job_type.Sphinx, job_name)
        job.set_occupancy_smearing(self.ekt_scheme, width=self.ekt)
        job.structure = structure
        positions = [p[2] for p in job.structure.positions]
        job.structure.add_tag(selective_dynamics=(True, True, True))
        job.structure.selective_dynamics[
            np.where(np.asarray(positions) < z_height)[0]
        ] = (False, False, False)
        if constrained:
            job.structure.selective_dynamics[index] = (False, False, True)
        job.set_kpoints(self.k_cut)
        job.set_encut(self.en_cut)
        job.set_convergence_precision(electronic_energy=self.e_energy, ionic_energy_tolerance=self.i_energy)
        job.calc_minimize()
        right_field = self.e_field / 51.4
        left_field = 0.0
        cell = job.structure.cell * self.ANGSTROM_TO_BOHR
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        total_charge = (right_field - left_field) * area / (4 * np.pi)
        sort_positions = np.sort(positions)
        job.input.sphinx.initialGuess.rho.charged = Group({})
        job.input.sphinx.initialGuess.rho.charged.charge = total_charge
        job.input.sphinx.initialGuess.rho.charged.z = sort_positions[-2] * self.ANGSTROM_TO_BOHR
        if vdw:
            job.input.sphinx.PAWHamiltonian.vdwCorrection = Group({})
            job.input.sphinx.PAWHamiltonian.vdwCorrection.method = "\"D2\""
        job.input.sphinx.PAWHamiltonian.nExcessElectrons = -total_charge
        job.input.sphinx.PAWHamiltonian.dipoleCorrection = True
        job.input.sphinx.PAWHamiltonian.zField = left_field * self.HARTREE_TO_EV
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag'][
            'rhoMixing'] = self.rhomixing  # using conservative mixing can help with convergence.
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'type'] = self.preconditioner
        job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
            'scaling'] = self.preconscaling
        job.fix_symmetry = False
        job.input['THREADS'] = self.threads
        queue = 'cm'
        job.server.cores = self.cores
        job.server.queue = queue
        return job

    @staticmethod
    def get_high_index_surface(element='Ni', crystal_structure='fcc', lattice_constant=3.526,
                               terrace_orientation=None, step_orientation=None, kink_orientation=None,
                               step_down_vector=None, length_step=3, length_terrace=3, length_kink=1, layers=60,
                               vacuum=10):
        """
        Gives the miller indices of high index surface required to create a stepped and kink surface, based on the
        general orientation and length of terrace, step and kinks respectively. The microfacet notation used is based
        on the work of Van Hove et al.,[1].
        Additionally also returns a bottom slab with the calculated high index surface
        [1] Van Hove, M. A., and G. A. Somorjai. "A new microfacet notation for high-Miller-index surfaces of cubic
        materials with terrace, step and kink structures." Surface Science 92.2-3 (1980): 489-518.
        Args:
            element (str): The parent element eq. "N", "O", "Mg" etc.
            crystal_structure (str): The crystal structure of the lattice
            lattice_constant (float): The lattice constant
            terrace_orientation (list): The miller index of the terrace eg., [1,1,1]
            step_orientation (list): The miller index of the step eg., [1,1,0]
            kink_orientation (list): The miller index of the kink eg., [1,1,1]
            step_down_vector (list): The direction for stepping down from the step to next terrace eg., [1,1,0]
            length_terrace (int): The length of the terrace along the kink direction in atoms eg., 3
            length_step (int): The length of the step along the step direction in atoms eg., 3
            length_kink (int): The length of the kink along the kink direction in atoms eg., 1
            layers (int): Number of layers of the high_index_surface eg., 60
            vacuum (float): Thickness of vacuum on the top of the slab
        Returns:
            high_index_surface: The high miller index surface which can be used to create slabs
            fin_kink_orientation: The kink orientation lying in the terrace
            fin_step_orientation: The step orientation lying in the terrace
            slab: pyiron_atomistics.atomistics.structure.atoms.Atoms instance Required surface
        """
        basis = CrystalStructure(name=element, crystalstructure=crystal_structure, a=lattice_constant)
        sym = basis.get_symmetry()
        eqvdirs = np.unique(np.matmul(sym.rotations[:], (np.array(step_orientation))), axis=0)
        eqvdirk = np.unique(np.matmul(sym.rotations[:], (np.array(kink_orientation))), axis=0)
        eqvdirs_ind = np.where(np.dot(np.squeeze(eqvdirs), terrace_orientation) == 0)[0]
        eqvdirk_ind = np.where(np.dot(np.squeeze(eqvdirk), terrace_orientation) == 0)[0]
        if len(eqvdirs_ind) == 0:
            raise ValueError('Step orientation vector should lie in terrace.\
            For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
            please change the stepOrientation and try again')
        if len(eqvdirk_ind) == 0:
            raise ValueError('Kink orientation vector should lie in terrace.\
            For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
            please change the kinkOrientation and try again')
        temp = (np.cross(np.squeeze(eqvdirk[eqvdirk_ind[0]]), np.squeeze(eqvdirs))).tolist().index(terrace_orientation)
        fin_kink_orientation = eqvdirk[eqvdirk_ind[0]]
        fin_step_orientation = eqvdirs[temp]
        vec1 = (np.asanyarray(fin_step_orientation).dot(length_step)) + \
               (np.asanyarray(fin_kink_orientation).dot(length_kink))
        vec2 = (np.asanyarray(fin_kink_orientation).dot(length_terrace)) + step_down_vector
        high_index_surface = np.cross(np.asanyarray(vec1), np.asanyarray(vec2))
        high_index_surface = np.array(high_index_surface / np.gcd.reduce(high_index_surface), dtype=int)
        surf = surface(basis, high_index_surface, layers, vacuum)
        sga = SpacegroupAnalyzer(pyiron_to_pymatgen(ase_to_pyiron(surf)))
        pmg_refined = sga.get_refined_structure()
        slab = pymatgen_to_pyiron(pmg_refined)
        slab.positions[:, 2] = slab.positions[:, 2] - np.min(slab.positions[:, 2])
        slab.set_pbc(True)
        slab.set_initial_magnetic_moments(np.repeat(1.0, len(slab)))
        return slab, high_index_surface, fin_kink_orientation, fin_step_orientation

    @staticmethod
    def get_slab(element='Ni', structure='fcc', a=3.526, layers=8, hkl=None, vac=10, mag_moms=True):
        """A function to create a slab with a given hkl index positioned at the bottom of the cell
        with vacuum on the top"""
        tb = bulk(element, structure, a=a, cubic=True)
        sa = surface(tb, hkl, layers, vacuum=vac)
        sap = ase_to_pyiron(sa)
        pmg_slab = pyiron_to_pymatgen(sap)
        sga = SpacegroupAnalyzer(pmg_slab)
        pmg_refined = sga.get_refined_structure()
        slab = pymatgen_to_pyiron(pmg_refined)
        slab.positions[:, 2] = slab.positions[:, 2] - np.min(slab.positions[:, 2])
        slab.set_pbc(True)
        if mag_moms is True:
            slab.set_initial_magnetic_moments(np.repeat(1.0, len(slab)))
        return slab

    @staticmethod
    def add_adsorbate_slab(structure, adsorbate='Ne', cut_off_volume=None):
        """Adds a mono adsorbate layer on the vacuum side of the stepped slab. Needs the adsorbate element,
        stepped slab as structure and a cut_off_volume (if None takes the median of voronoi volume as cutoff) based
        on voronoi volume to identify surface """
        if cut_off_volume is None:
            surf_ind = np.where(structure.analyse.pyscal_voronoi_volume() >
                                np.median(structure.analyse.pyscal_voronoi_volume()))
        else:
            surf_ind = np.where(structure.analyse.pyscal_voronoi_volume() > cut_off_volume)
        adsorbed_structure = structure.to_ase()
        for i in range(len(surf_ind[0])):
            if structure.positions[surf_ind[0][i], 2] > np.mean(structure.positions[:, 2]):
                add_adsorbate(adsorbed_structure, adsorbate,
                              position=(structure.positions[surf_ind[0][i], 0], structure.positions[surf_ind[0][i], 1]),
                              height=1.5)
        return ase_to_pyiron(adsorbed_structure)

    @staticmethod
    def add_adsorbate_slab_positions(structure, adsorbate_positions, adsorbate='Ne', adsorbate_height=1.5):
        """Adds a mono adsorbate layer based on user defined positions. Needs the adsorbate element,
        slab structure, positions on which adsorbate has to be added, and adsorbate height """
        adsorbed_structure = structure.to_ase()
        for i in range(len(adsorbate_positions)):
            add_adsorbate(adsorbed_structure, adsorbate,
                          position=(structure.positions[adsorbate_positions[i], 0],
                                    structure.positions[adsorbate_positions[i], 1]),
                          height=adsorbate_height)
        return ase_to_pyiron(adsorbed_structure)

    @staticmethod
    def hirshfeld_charges(job):
        """
        Hirshfeld job analysis for SPHInX jobs.
        :param job: SPHInX jobs
        :return: hc is an array of hirshfeld charges
        """
        job.decompress()
        subprocess.call([f'cd {job.working_directory};module load sphinx;sxpawatomvolume --log'], shell=True)
        with open(job.working_directory + '/sxpawatomvolume.log', 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            f.close()
        hcl = last_line.split(':')[1].strip('[];')[2:].split(',')
        hc = np.array(hcl, dtype=np.float32)
        return hc

    def restart_gdc_calculations(self, new_job_name, old_job_name, TS=False):
        """Restart evaporation or relaxation or TS generalized dipole correction calculations  from previous charge
        density of old_job_name
        :param TS: set to True if restarting calculation is transition stateoptimization
        :param old_job_name: name of the job to restart
        :param new_job_name: name of the restarting job
        :returns restarted pyiron job
        """
        old_job = self.pr.load(old_job_name, convert_to_object=True)
        charge_density_file = old_job.working_directory + '/rho.sxb'
        job = self.pr.create_job(self.pr.job_type.Sphinx, new_job_name)
        job.input = old_job.input
        job.input.sphinx.initialGuess.rho = Group({"file": '"' + charge_density_file + '"'})
        job.structure = old_job.get_structure(-1)
        if TS:
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag'][
                'rhoMixing'] = self.rhomixing  # using conservative mixing can help with convergence.
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag']['preconditioner'][
                'type'] = self.preconditioner
            job.input['sphinx']['main']['ricTS']['bornOppenheimer']['scfDiag']['preconditioner'][
                'scaling'] = self.preconscaling
            job.executable.version = '3.0'
        else:
            job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['rhoMixing'] = self.rhomixing
            job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
                'type'] = self.preconditioner
            job.input['sphinx']['main']['ricQN']['bornOppenheimer']['scfDiag']['preconditioner'][
                'scaling'] = self.preconscaling
        queue = 'cm'
        job.server.cores = self.cores
        job.server.queue = queue
        job.input['THREADS'] = self.threads
        return job
