X-Ray Timing Made Easy
=======================
+------------------+-------------------------+----------------------------+
| Master           | |Build Status Master|   | |Coverage Status Master|   |
+------------------+-------------------------+----------------------------+

We are writing a general-purpose timing package for X-ray time series.

Note to Users
-------------

Stingray is currently in **development phase**. Some of the code is not in
its final stage, and might change before our first release. There might also
still be bugs we are working on fixing and features that are not finished.

We encourage you to download it and try it out, but please be aware of
the caveats of working with in-development code.
At the same time, we welcome contributions and we need your help!
If you have your own code duplicating any part of the methods implemented in
Stingray, please try out Stingray and compare to your own results.

We do welcome any sort of feedback: if something breaks, please report it via
the `issues<https://github.com/dhuppenkothen/stingray/issues>`_. Similarly,
please open an issue if any functionality is missing, the API is not intuitive
or if you have suggestions for additional functionality that would be useful to
have.

If you have code you might want to contribute, we'd love to hear from you,
either via a `pull request<https://github.com/StingraySoftware/stingray/pulls>`_
or via an `issue<https://github.com/dhuppenkothen/stingray/issues>`_.

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

Documentation
-------------

Is generated using `Sphinx`_. Try::

   $ sphinx-build docs docs/_build

Then open ``./docs/_build/index.html`` in the browser of your choice.

.. _Sphinx: http://sphinx-doc.org

Test suite
----------

Try::

   $ nosetests

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
