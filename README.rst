X-Ray Spectral Timing Made Easy
===============================
+------------------+-------------------------+----------------------------+
| Master           | |Build Status Master|   | |Coverage Status Master|   |
+------------------+-------------------------+----------------------------+

Stingray is an in-development spectral-timing software package for astrophysical X-ray (and more) data.
Stingray merges existing efforts for a (spectral-)timing package in Python, and is 
structured with the best guidelines for modern open-source programming, following the example of `Astropy`_ .

It is composed of:

1. a library of time series methods, including power spectra, cross spectra, covariance spectra, lags, and so on; 
2. a set of scripts to load FITS data files from different missions;
3. a simulator of light curves and event lists, that includes different kinds of variability and more complicated phenomena based on the impulse response of given physical events (e.g. reverberation);
4. finally, an in-development GUI to ease the learning curve for new users.

There are a number of official software packages for X-ray spectral fitting (Xspec, ISIS, Sherpa, ...). 
Such a widely used and standard software package does not exist for X-ray timing, 
that remains for now mostly done with custom software. 
Stingray aims not only at becoming a standard timing package, 
but at extending the implementation to the most advanced spectral timing techniques available in the literature. 
The ultimate goal of this project is to provide the community with a package that eases 
the learning curve for the advanced spectral timing techniques with a correct statistical framework.

Contents
--------

The contents will be:

- make a light curve from event data
- make periodograms in Leahy and rms normalization
- average periodograms
- cross spectra and lags (time vs energy, time vs frequency)
- maximum likelihood fitting of periodograms/parametric models
- bispectra (?)
- cross correlation functions, coherence
- spectral-timing functionality
- Bayesian QPO searches
- power colours
- rms spectra

Installation
-------------

1. Clone our project from github (remember about including --recursive flag due to our submodule(s))::

    $ git clone --recursive https://github.com/StingraySoftware/stingray.git

2. Go to already created directory and install all dependencies::

    $ pip install -r requirements.txt

3. Go back to stingray project directory and execute::

    $ python setup.py install


Documentation
-------------

Is generated using `Sphinx`_. Try::

   $ sphinx-build docs docs/_build

Then open ``./docs/_build/index.html`` in the browser of your choice.

.. _Sphinx: http://sphinx-doc.org

Test suite
----------

Stingray uses `py.test` for testing. To run the tests, try::

   $ python setup.py test 

Copyright
---------

All content © 2015 the authors. The code is distributed under the MIT license.

Pull requests are welcome! If you are interested in the further development of
this project, please `get in touch via the issues
<https://github.com/dhuppenkothen/stingray/issues>`_!

.. |Build Status Master| image:: https://travis-ci.org/StingraySoftware/stingray.svg?branch=master
    :target: https://travis-ci.org/StingraySoftware/stingray
.. |Coverage Status Master| image:: https://coveralls.io/repos/github/StingraySoftware/stingray/badge.svg?branch=master
    :target: https://coveralls.io/github/StingraySoftware/stingray?branch=master
.. _Astropy: https://www.github.com/astropy/astropy
