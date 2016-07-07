
import numpy as np
import os

from ..events import EventList
from ..lightcurve import Lightcurve

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False

class TestEvents(object):
	
	@classmethod
	def setup_class(self):
		self.time = [0.5, 1.5, 2.5, 3.5]
		self.counts = [3000, 2000, 2200, 3600]
		self.spectrum = [[1, 2, 3, 4, 5, 6],[1000, 2040, 1000, 3000, 4020, 2070]]

	def test_to_lc(self):
		"""
		Create a light curve from event list
		"""
		ev = EventList(self.time)
		lc = ev.to_lc(0.5)

	def test_set_times(self):
		"""
		Set photon arrival times for an event list 
		from light curve.
		"""
		lc = Lightcurve(self.time, self.counts)
		ev = EventList(self.time)
		ev.set_times(lc)

	def test_set_energies(self):
		"""
		Assign photon energies to an event list.
		"""
		ev = EventList(self.time)
		ev.set_energies(self.spectrum)

	def test_io_with_ascii(self):
		"""
		Test IO methods with ascii format.
		"""
		ev = EventList(self.time)
		ev.write('ascii_ev.txt',format_='ascii')
		ev.read('ascii_ev.txt', format_='ascii')
		os.remove('ascii_ev.txt')

	def test_io_with_pickle(self):
		"""
		Test IO methods with pickle format.
		"""
		ev = EventList(self.time)
		ev.write('ev.pickle', format_='pickle')
		ev.read('ev.pickle',format_='pickle')
		assert np.all(ev.time == self.time)
		os.remove('ev.pickle')

	def test_io_with_hdf5(self):
		"""
		Test IO methods with hdf5 format.
		"""
		ev = EventList(time=self.time)
		ev.write('ev.hdf5', format_='hdf5')

		if _H5PY_INSTALLED:
			data = ev.read('ev.hdf5',format_='hdf5')
			assert np.all(data['time'] == self.time)
			os.remove('ev.hdf5')

		else:
			ev.read('ev.pickle',format_='pickle')
			assert np.all(ev.time == self.time)
			os.remove('ev.pickle')

