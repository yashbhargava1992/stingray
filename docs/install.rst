Downloading and Installing Stingray
===================================

There are currently two ways to install Stingray:

* via ``pip``
* from source

A ``conda`` package is currently in the works, but not available yet.

Installing via ``pip``
----------------------

``pip``-installing Stingray is easy! Just do::

    $ pip install stingray

And you should be done! Just remember to run the tests before you use it!

Installing from source (bleeding edge version)
----------------------------------------------

For those of you wanting to install the bleeding-edge development version from
source (it *will* have bugs; you've been warned!), first clone
`our repository <https://github.com/StingraySoftware/stingray>`_ on GitHub: ::

    $ git clone --recursive https://github.com/StingraySoftware/stingray.git

Now ``cd`` into the newly created ``stingray`` directory and install the necessary
dependencies: ::

    $ cd stingray
    $ pip install -r requirements.txt

Finally, install ``stingray`` itself: ::

    $ python setup.py install

Test Suite
----------

Please be sure to run the test suite before you use the package, and please report anything
you think might be bugs on our GitHub `Issues page <https://github.com/StingraySoftware/stingray/issues>`_.

Stingray uses `py.test <https://doc.pytest.org/en/latest/>`_ for testing. To run the tests, go into
the ``stingray`` root directory and execute ::

    $ python setup.py test

Documentation
-------------

The documentation including tutorials is hosted on `readthedocs <https://stingray.readthedocs.io>`_
The documentation uses `sphinx <http://www.sphinx-doc.org/en/stable/>`_ to build and requires a couple
of extensions (most notably `nbsphinx <http://nbsphinx.readthedocs.io/en/0.3.1/>`_ and the
`astropy helpers <https://github.com/astropy/astropy-helpers>`_).

You can build the API reference yourself by going into the ``docs`` folder within the ``stingray`` root
directory and running the ``Makefile``: ::

    $ cd stingray/docs
    $ make html

If that doesn't work on your system, you can invoke ``sphinx-build`` itself from the stingray source directory: ::

    $ cd stingray
    $ $ sphinx-build docs docs/_build

The documentation should be located in ``stingray/docs/_build``. Try opening ``./docs/_build/index.rst`` from
the stingray source directory.