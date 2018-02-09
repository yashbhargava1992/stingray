Downloading and Installing Stingray
===================================

There are currently two ways to install Stingray:
* via ``pip``
* from source

A ``conda`` package is currently in the works, but not available yet.

Installing via ``pip``
----------------------

``pip``-installing Stingray is easy! Just do

..

    $ pip install stingray


And you should be done! Just remember to run the tests before you use it!

Installing from source (bleeding edge version)
----------------------------------------------

For those of you wanting to install the bleeding-edge development version from
source (it *will* have bugs; you've been warned!), first clone our repository on
GitHub:

..

    $ git clone --recursive https://github.com/StingraySoftware/stingray.git


Now ``cd`` into the newly created ``stingray`` directory and install the necessary
dependencies:

..

    $ pip install -r requirements.txt

Finally, install ``stingray`` itself:

..

    $ python setup.py install

Test Suite
----------

Please be sure to run the test suite before you use the package, and please report anything
you think might be bugs on our GitHub `Issues page <https://github.com/StingraySoftware/stingray/issues>`_.

Stingray uses ```py.test`` <https://doc.pytest.org/en/latest/>`_ for testing. To run the tests, go into
the ``stingray`` root directory and execute

..

    $ python setup.py test

Documentation
-------------

