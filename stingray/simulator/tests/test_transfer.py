from stingray.simulator import transfer
import pytest

class TestSimulator(object):

    @classmethod
    def setup_class(self):
        arr = [[0 for j in range(5)] for i in range(10)]
        self.transfer = transfer.TransferFunction(arr)

    def test_incorrect_rows(self):
        """Test if exception is raised in case there is 1
        or no row.
        """
        arr = [[0 for j in range(5)] for i in range(1)]
        with pytest.raises(ValueError):
            transfer.TransferFunction(arr)

    def test_incorrect_columns(self):
        """Test if exception is raised in case there is 1
        or no column.
        """
        arr = [[0 for j in range(1)] for i in range(10)]
        with pytest.raises(ValueError):
            transfer.TransferFunction(arr)

    def test_time_response(self):
        """Test obtaining a time-resolved response."""
        self.transfer.time_response()

    def test_average_energy(self):
        """Test obtaining an energy-resolved response."""
        self.transfer.energy_response()

    def test_plot(self):
        """Test plotting a transfer function."""
        self.transfer.plot()

    def test_read(self):
        self.transfer.read()

    def test_write(self):
        self.transfer.write()

    def test_simple_ir(self):
        """Test constructing a simple impulse response."""
        t0, w = 100, 500
        assert len(transfer.simple_ir(1, t0, w)), (t0+w)

    def test_relativistic_ir(self):
        """
        Test constructing a relativistic impulse response."""
        t1, t3 = 3, 10
        assert len(transfer.relativistic_ir(1, t1=t1, t3=t3)), (t1+t3)



