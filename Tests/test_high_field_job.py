import unittest
from HighElectricField import HighFieldModule
from pyiron import Project


class TestHighFieldModule(unittest.TestCase):
    def test_high_index_surface(self):
        pr = Project("Tests")
        structure = pr.create_structure('Ni', 'fcc', 3.526)
        e_field = 3.0
        encut = 500
        kcut = [6, 6, 1]
        job_name = 'test'
        tester = HighFieldModule.HighFieldJob(pr=pr, job_name=job_name,
                                              structure=structure, e_field=e_field,
                                              encut=encut, kcut=kcut)
        slab, h, s, k = tester.get_high_index_surface(element='Ni',
                                                      crystal_structure='fcc',
                                                      lattice_constant=3.526,
                                                      terrace_orientation=[1, 1, 1],
                                                      step_orientation=[1, 1, 0],
                                                      kink_orientation=[1, 0, 1],
                                                      step_down_vector=[1, 1, 0],
                                                      length_step=2,
                                                      length_terrace=3,
                                                      length_kink=1, layers=60,
                                                      vacuum=10)
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0], -9)
        self.assertEqual(len(k), 3)
        self.assertEqual(len(s), 3)

    def test_get_slab(self):
        pr = Project("Tests")
        structure = pr.create_structure('Ni', 'fcc', 3.526)
        e_field = 3.0
        encut = 500
        kcut = [6, 6, 1]
        job_name = 'test'
        tester = HighFieldModule.HighFieldJob(pr=pr, job_name=job_name,
                                              structure=structure, e_field=e_field,
                                              encut=encut, kcut=kcut)
        slab = tester.get_slab(element='Ni', structure='fcc', a=3.526,
                               layers=8, hkl=[2, 1, 1], vac=10, mag_moms=True)

        self.assertEqual(len(slab),16)


if __name__ == '__main__':
    unittest.main()
