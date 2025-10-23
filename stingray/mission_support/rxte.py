import re
import numpy as np
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy.table import Table

from astropy.io import fits
from stingray.io import fits_open_including_remote

c_match = re.compile(r"C\[(.*)\]")

_EDGE_TIMES = [
    "1995-12-30T00:00:00.0",  # launch
    "1996-03-21T18:33:00.0",
    "1996-04-15T23:05:00.0",
    "1999-03-22T17:37:00.0",
    "2000-05-13T00:00:00.0",
    "2012-01-05T00:00:00.0",  # decommissioning
]

_EDGE_EPOCHS = Time(_EDGE_TIMES, format="isot", scale="utc").mjd


def _split_chan_spec(chan, sep="~"):
    """Split a channel specification into a tuple of integers.

    If the specification is, e.g., ``10-13``, it will be interpreted as
    ``(10, 13)``. The same thing will be omitted if the starting digit(s) of the first int are
    omitted, e.g. ``10-3`` will still be interpreted as ``(10, 13)``.
    If there is only one integer, it will be duplicated.

    Examples
    --------
    >>> _split_chan_spec("10~13")
    (10, 13)
    >>> _split_chan_spec("10-3", sep="-")
    (10, 13)
    >>> _split_chan_spec("10")
    (10, 10)

    """
    if sep in chan:
        cs = chan.split(sep)
    else:
        cs = (chan, chan)
    if len(cs[1]) < len(cs[0]):
        c0 = cs[0]
        c1 = cs[0][: -len(cs[1])] + cs[1]
        cs = (c0, c1)
    return (int(cs[0]), int(cs[1]))


def _split_C_string(string, sep="~"):
    """Interpret the C string in the TEVTB2 key and return a list of channel tuples."""
    channel_list = []
    if ":" in string:
        cs_range = string.split(":")
        return [(c, c) for c in range(int(cs_range[0]), int(cs_range[1]) + 1)]

    for chan in string.split(","):
        cs = _split_chan_spec(chan, sep)
        channel_list.append(cs)
    return channel_list


def _decode_energy_channels(tevtb2):
    """Understand the channel information TEVTB2 key.

    Parameters
    ----------
    tevtb2 : str
        The TEVTB2 key from the FITS header.

    Returns
    -------
    chans: list of tuples
        A list of tuples, each containing the start and stop channel of a group of channels.

    Examples
    --------
    >>> tevtb2 = '(M[1]{1},C[43,44~45]{6})'
    >>> chans = _decode_energy_channels(tevtb2)
    >>> assert chans == [(43, 43), (44, 45)]
    >>> _decode_energy_channels('(M[1]{1})')
    Traceback (most recent call last):
    ...
    ValueError: No C line found in the TEVTB2 key.
    """
    dll_fmt_string_split = re.split(",(?=[A-Z])", tevtb2)
    for line in dll_fmt_string_split:
        if not line.startswith("C"):
            continue
        line = c_match.match(line).group(1)
        break
    else:
        raise ValueError("No C line found in the TEVTB2 key.")

    return _split_C_string(line)


def pca_calibration_func(epoch):
    """Return the appropriate calibration function for RXTE for a given observing epoch.

    This function has signature ``func(pha, detector_id)`` and gives the energy corresponding
    to the PHA channel for the given detector (array values allowed).

    Internally, this is done by pre-allocating some arrays with the energy values for each
    PHA channel and detector group (1-4 and 0, due to a damage that PCU 0 incurred in 2000),
    and then returning a function that looks up the energy for each channel.

    This does not require any interpolation, as the calibration is tabulated for each channel,
    and it is pretty efficient given the very small number of channels supported by the PCA (255).

    Parameters
    ----------
    epoch : float
        The epoch of the observation in MJD.

    Returns
    -------
    conversion_function : callable
        A function that converts PHA channel to energy. This function accepts
        two arguments: the PHA channel and the PCU number.

    Examples
    --------
    >>> conversion_function = pca_calibration_func(50082)
    >>> float(conversion_function(10, 0))
    3.04
    >>> conversion_function = pca_calibration_func(55930)
    >>> float(conversion_function(10, 0))
    4.53
    >>> float(conversion_function(10, 3))
    4.49
    >>> assert np.array_equal(conversion_function(10, [0, 3]), [4.53, 4.49])
    >>> assert np.array_equal(conversion_function([10, 11], [0, 3]), [4.53, 4.90])
    """
    caltable = Table.read(
        """
    Abs   STD2  E1   E2    E3   E4   E5_0   E5_1234
    0-4	0	1.51	1.61	1.94	2.13	1.95	  2.06
    5	1	1.76	1.91	2.29	2.54	2.38	  2.47
    6	2	2.02	2.22	2.64	2.96	2.81	  2.87
    7	3	2.27	2.52	2.99	3.37	3.24	  3.28
    8	4	2.53	2.83	3.35	3.79	3.67	  3.68
    9	5	2.78	3.13	3.70	4.21	4.09	  4.09
    10	6	3.04	3.44	4.05	4.63	4.53	  4.49
    11	7	3.30	3.75	4.41	5.04	4.96	  4.90
    12	8	3.55	4.05	4.76	5.46	5.39	  5.31
    13	9	3.81	4.36	5.12	5.88	5.82	  5.71
    14	10	4.07	4.67	5.47	6.30	6.25	  6.12
    15	11	4.32	4.98	5.82	6.72	6.68	  6.53
    16	12	4.58	5.28	6.18	7.14	7.11	  6.94
    17	13	4.84	5.59	6.54	7.56	7.55	  7.35
    18	14	5.10	5.90	6.89	7.98	7.98	  7.76
    19	15	5.35	6.21	7.25	8.40	8.42	  8.17
    20	16	5.61	6.52	7.60	8.82	8.85	  8.57
    21	17	5.87	6.83	7.96	9.24	9.28	  8.98
    22	18	6.13	7.14	8.32	9.67	9.72	  9.40
    23	19	6.39	7.45	8.68	10.09	10.16	  9.81
    24	20	6.65	7.76	9.03	10.51	10.59	  10.22
    25	21	6.91	8.07	9.39	10.93	11.03	  10.63
    26	22	7.17	8.38	9.75	11.36	11.47	  11.04
    27	23	7.43	8.69	10.11	11.78	11.90	  11.45
    28	24	7.69	9.00	10.47	12.21	12.34	  11.87
    29	25	7.95	9.31	10.83	12.63	12.78	  12.28
    30	26	8.21	9.63	11.19	13.06	13.22	  12.69
    31	27	8.47	9.94	11.55	13.48	13.66	  13.11
    32	28	8.73	10.25	11.91	13.91	14.10	  13.52
    33	29	8.99	10.56	12.27	14.34	14.54	  13.93
    34	30	9.25	10.88	12.63	14.76	14.98	  14.35
    35	31	9.52	11.19	12.99	15.19	15.42	  14.76
    36	32	9.78	11.50	13.36	15.62	15.86	  15.18
    37	33	10.04	11.82	13.72	16.05	16.30	  15.60
    38	34	10.30	12.13	14.08	16.47	16.74	  16.01
    39	35	10.57	12.45	14.44	16.90	17.19	  16.43
    40	36	10.83	12.76	14.81	17.33	17.63	  16.85
    41	37	11.09	13.08	15.17	17.76	18.07	  17.26
    42	38	11.36	13.39	15.54	18.19	18.52	  17.68
    43	39	11.62	13.71	15.90	18.62	18.96	  18.10
    44	40	11.89	14.03	16.26	19.05	19.41	  18.52
    45	41	12.15	14.34	16.63	19.49	19.85	  18.94
    46	42	12.42	14.66	17.00	19.92	20.30	  19.36
    47	43	12.68	14.98	17.36	20.35	20.75	  19.78
    48	44	12.95	15.29	17.73	20.78	21.19	  20.20
    49	45	13.21	15.61	18.09	21.22	21.64	  20.62
    50	46	13.48	15.93	18.46	21.65	22.09	  21.04
    51	47	13.74	16.25	18.83	22.08	22.54	  21.46
    52	48	14.01	16.57	19.20	22.52	22.98	  21.88
    53	49	14.28	16.89	19.56	22.95	23.43	  22.30
    54-5	50	14.81	17.52	20.30	23.82	24.33	  23.15
    56-7	51	15.35	18.16	21.04	24.70	25.24	  24.00
    58-9	52	15.88	18.81	21.78	25.57	26.14	  24.85
    60-1	53	16.42	19.45	22.52	26.45	27.05	  25.70
    62-3	54	16.96	20.09	23.26	27.33	27.95	  26.55
    64-5	55	17.50	20.74	24.01	28.21	28.86	  27.40
    66-7	56	18.04	21.39	24.75	29.09	29.78	  28.26
    68-9	57	18.58	22.03	25.50	29.97	30.69	  29.12
    70-1	58	19.12	22.68	26.25	30.86	31.61	  29.97
    72-3	59	19.67	23.33	27.00	31.74	32.52	  30.83
    74-5	60	20.21	23.99	27.75	32.63	33.44	  31.70
    76-7	61	20.76	24.64	28.50	33.52	34.36	  32.56
    78-9	62	21.30	25.29	29.26	34.42	35.29	  33.43
    80-1	63	21.85	25.95	30.02	35.31	36.21	  34.29
    82-3	64	22.40	26.61	30.77	36.21	37.14	  35.16
    84-5	65	22.95	27.27	31.53	37.10	38.07	  36.03
    86-7	66	23.50	27.93	32.29	38.00	39.01	  36.91
    88-9	67	24.06	28.59	33.06	38.91	39.94	  37.78
    90-1	68	24.61	29.25	33.82	39.81	40.88	  38.66
    92-3	69	25.17	29.91	34.59	40.72	41.82	  39.53
    94-5	70	25.72	30.58	35.35	41.62	42.76	  40.41
    96-7	71	26.28	31.25	36.12	42.53	43.70	  41.30
    98-9	72	26.84	31.92	36.89	43.44	44.64	  42.18
    100-1	73	27.40	32.59	37.67	44.36	45.59	  43.06
    102-3	74	27.96	33.26	38.44	45.27	46.54	  43.95
    104-5	75	28.52	33.93	39.22	46.19	47.49	  44.84
    106-7	76	29.09	34.61	39.99	47.11	48.45	  45.73
    108-9	77	29.65	35.28	40.77	48.03	49.41	  46.62
    110-1	78	30.22	35.96	41.55	48.96	50.36	  47.52
    112-3	79	30.79	36.64	42.34	49.88	51.33	  48.41
    114-5	80	31.36	37.32	43.12	50.81	52.29	  49.31
    116-7	81	31.93	38.00	43.91	51.74	53.25	  50.21
    118-9	82	32.50	38.68	44.70	52.67	54.22	  51.12
    120-1	83	33.07	39.37	45.49	53.61	55.19	  52.02
    122-3	84	33.64	40.06	46.28	54.54	56.17	  52.93
    124-5	85	34.22	40.74	47.07	55.48	57.14	  53.83
    126-7	86	34.80	41.43	47.87	56.42	58.12	  54.74
    128-9	87	35.38	42.13	48.66	57.37	59.10	  55.66
    130-1	88	35.95	42.82	49.46	58.31	60.08	  56.57
    132-3	89	36.54	43.51	50.26	59.26	61.07	  57.49
    134-5	90	37.12	44.21	51.06	60.21	62.06	  58.40
    136-8	91	37.99	45.26	52.27	61.64	63.54	  59.78
    139-41	92	38.87	46.31	53.48	63.07	65.04	  61.17
    142-4	93	39.75	47.36	54.70	64.51	66.54	  62.56
    145-7	94	40.64	48.42	55.92	65.95	68.04	  63.96
    148-50	95	41.53	49.49	57.14	67.40	69.55	  65.36
    151-3	96	42.42	50.55	58.37	68.86	71.07	  66.76
    154-6	97	43.32	51.62	59.60	70.32	72.59	  68.17
    157-9	98	44.21	52.70	60.84	71.78	74.12	  69.59
    160-2	99	45.12	53.78	62.08	73.25	75.65	  71.01
    163-5	100	46.02	54.86	63.33	74.73	77.19	  72.43
    166-8	101	46.93	55.95	64.58	76.21	78.74	  73.86
    169-71	102	47.84	57.04	65.84	77.70	80.30	  75.30
    172-4	103	48.76	58.13	67.10	79.19	81.86	  76.74
    175-7	104	49.68	59.23	68.37	80.69	83.43	  78.18
    178-80	105	50.60	60.33	69.64	82.20	85.00	  79.63
    181-3	106	51.53	61.44	70.91	83.71	86.58	  81.09
    184-6	107	52.46	62.55	72.19	85.23	88.17	  82.55
    187-9	108	53.39	63.67	73.48	86.75	89.76	  84.02
    190-2	109	54.33	64.79	74.77	88.28	91.37	  85.49
    193-5	110	55.27	65.92	76.07	89.81	92.98	  86.97
    196-8	111	56.22	67.05	77.37	91.36	94.59	  88.46
    199-201	112	57.17	68.18	78.68	92.91	96.22	  89.95
    202-4	113	58.12	69.32	79.99	94.46	97.85	  91.45
    205-7	114	59.08	70.47	81.30	96.02	99.49	  92.95
    208-10	115	60.04	71.62	82.63	97.59	101.14	  94.46
    211-3	116	61.00	72.77	83.96	99.17	102.79	  95.97
    214-6	117	61.97	73.93	85.29	100.75	104.46	  97.49
    217-9	118	62.95	75.10	86.63	102.34	106.13	  99.02
    220-2	119	63.93	76.27	87.98	103.93	107.81	  100.55
    223-5	120	64.91	77.44	89.33	105.54	109.50	  102.09
    226-8	121	65.90	78.62	90.69	107.15	111.19	  103.64
    229-31	122	66.89	79.81	92.05	108.76	112.90	  105.19
    232-4	123	67.89	81.00	93.42	110.39	114.61	  106.75
    235-7	124	68.89	82.20	94.80	112.02	116.33	  108.32
    238-41	125	70.23	83.80	96.64	114.21	118.65	  110.42
    242-5	126	71.58	85.42	98.50	116.41	120.97	  112.53
    246-9	127	72.94	87.04	100.37	118.63	123.32	  114.65
    250-5	128	74.99	89.50	103.19	121.98	126.87	  117.86""",
        format="ascii",
    )
    abs_chan = caltable["Abs"]
    chans = [_split_chan_spec(chan, sep="-") for chan in abs_chan]

    col_idx = np.searchsorted(_EDGE_EPOCHS, epoch)
    if col_idx == 5:
        col_1234 = "E5_1234"
        col_0 = "E5_0"
    else:
        col_1234 = col_0 = f"E{col_idx}"

    # Create a step function for each group of PCUs
    energies_0 = np.zeros(256)
    energies_1234 = np.zeros(256)
    for chan_sep, energy_0, energy_1234 in zip(chans, caltable[col_0], caltable[col_1234]):
        energies_0[chan_sep[0] : chan_sep[1] + 1] = energy_0
        energies_1234[chan_sep[0] : chan_sep[1] + 1] = energy_1234

    def func(chan, detector_id=0):
        if detector_id == 0:
            return energies_0[int(chan)]
        return energies_1234[int(chan)]

    return np.vectorize(func)


def rxte_calibration_func(instrument, epoch):
    """Return the calibration function for RXTE at a given epoch.

    Examples
    --------
    >>> calibration_func = rxte_calibration_func("PCa", 50082)
    >>> assert calibration_func(10) == pca_calibration_func(50082)(10)
    >>> rxte_calibration_func("HEXTE", 55930)
    Traceback (most recent call last):
    ...
    ValueError: Unknown XTE instrument: HEXTE
    """
    if instrument.lower() == "pca":
        return pca_calibration_func(epoch)
    raise ValueError(f"Unknown XTE instrument: {instrument}")


def rxte_pca_event_file_interpretation(input_data, header=None, hduname=None):
    """Interpret the FITS header of an RXTE event file.

    At the moment, only science event files are supported. In these files,
    the energy channels are stored in a column named PHA. However, this is not
    the PHA column that can be directly used to convert to energy. These are
    channels that get changed on a per-observation basis, and can be converted
    to the "absolute" PHA channels (the ones tabulated in `pca_calibration_func`)
    by using the TEVTB2 keyword. This function changes the content of the PHA column by
    putting in the mean "absolute" PHA channel corresponding to each local PHA
    channel.

    Parameters
    ----------
    input_data : str, fits.HDUList, fits.HDU, np.array
        The name of the FITS file to, or the HDUList inside, or the HDU with
        the data, or the data.

    Other parameters
    ----------------
    header : `fits.Header`, optional
        Compulsory if ``hdulist`` is not a class:`fits._BaseHDU`, a
        :class:`fits.HDUList`, or a file name. The header of the relevant extension.
    hduname : str, optional
        Name of the HDU (only relevant if hdulist is a :class:`fits.HDUList`),
        ignored otherwise.

    """
    if isinstance(header, str):
        header = fits.Header.fromstring(header)

    if isinstance(input_data, str):
        return rxte_pca_event_file_interpretation(
            fits_open_including_remote(input_data), header=header, hduname=hduname
        )

    if isinstance(input_data, fits.HDUList):
        if hduname is None and "XTE_SE" not in input_data:
            raise ValueError(
                "No XTE_SE extension found. At the moment, only science events "
                "are supported by Stingray for XTE."
            )
        if hduname is None:
            hduname = "XTE_SE"
        new_hdu = rxte_pca_event_file_interpretation(input_data[hduname], header=header)
        input_data[hduname] = new_hdu
        return input_data

    if isinstance(input_data, fits.hdu.base._BaseHDU):
        if header is None:
            header = input_data.header
        input_data.data = rxte_pca_event_file_interpretation(input_data.data, header=header)
        return input_data

    data = input_data
    if header is None:
        raise ValueError(
            "If the input data is not a HDUList or a HDU, the header must be specified"
        )

    tevtb2 = header["TEVTB2"]
    local_chans = np.asarray([int(np.mean(ch)) for ch in _decode_energy_channels(tevtb2)])

    data["PHA"] = local_chans[data["PHA"]]

    return data
