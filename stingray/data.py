
import csv
import pkg_resources

import stingray.lightcurve as lightcurve

def sample_data():

    """
    Import data from csv and return a light curve object.

    Returns
    -------
    sample: :class:`Lightcurve` object
        The :class:`Lightcurve` object with the desired time stamps
        and counts.
    """

    lc_file = pkg_resources.resource_stream(__name__,"datasets/lc_sample.csv")
    lc_reader = csv.reader(lc_file)

    # Import first and second columns. While other may be useful, these are
    # sufficient for the purpose of generating a sample light curve.

    # Round the data to first decimal point. This will be generate a 24.0 hour
    # bin.

    data = [(round(float(row[0]),1),float(row[1])) for row in lc_reader]

    # Unpack the data
    dates = [i[0] for i in data]
    counts = [i[1] for i in data]

    # Return class:`Lightcurve` object
    return lightcurve.Lightcurve(dates, counts)
     
