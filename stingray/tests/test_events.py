
from ..events import EventList
from ..lightcurve import Lightcurve

class TestEvents(object):
	
	@classmethod
	def setup_class(self):
		self.times = [0.5, 1.5, 2.5, 3.5]
		self.counts = [3000, 2000, 2200, 3600]
		self.spectrum = [[1, 2, 3, 4, 5, 6],[1000, 2040, 1000, 3000, 4020, 2070]]

	def test_from_lc(self):
		"""
		Create an event list from light curve.
		"""
		lc = Lightcurve(self.times, self.counts)
		events = EventList(lc)

	def test_to_lc(self):
		"""
		Create a light curve from event list
		"""
		lc = EventList(self.spectrum)

	def test_assign_energies(self):
		"""
		Assign energies to an event list.
		"""
		lc = Lightcurve(self.times, self.counts)
		events = EventList(lc)
		events.energies(self.spectrum)

