from stingray.simulator import transfer

class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.transfer = transfer.TransferFunction([])
        self.transfer.average_time()
        self.transfer.average_energy()

    def test_simple_ir(self):
        """
        Test constructing a simple impulse response.
        """
        t0, w = 100, 500
        assert len(transfer.simple_ir(1, t0, w)), (t0+w)

    def test_relativistic_ir(self):
        """
        Test constructing a relativistic impulse response.
        """
        t1, t3 = 3, 10
        assert len(transfer.relativistic_ir(1, t1=t1, t3=t3)), (t1+t3)
