Download and install
--------------------------

This section covers the necessary steps to install and run SNPknock. Below
is a list of required dependencies, along with additional software
recommendations. 
The following instructions are for Linux and Mac users. 
It may be possible to compile the sources under Windows, but we are
not providing support for Windows.

SNPknock is currently under development and it should be considered to be 
of *beta* quality. Any feedback will be greatly appreciated.


Dependencies
^^^^^^^^^^^^

  **Python** 2.7.5 or later
  
  **NumPy** 1.13.1 or later
    A high-level, optimized scientific computing Python library.

  **Cython** 0.26 or later
    Cython is a language that is a fusion of Python and C.  It allows us
    to write fast code using Python and C syntax, so that it easier to
    read and maintain. You need Cython if you are building the
    development source code (and that is what you have to do at the
    moment, because we don't yet have a release).

  **gcc** 5.3.0 or later
    The GNU Compiler Collection (GCC) is a compiler system produced by the 
    GNU Project supporting various programming languages. Here, it is needed
    to compile the core of the SNPknock source code that is written in C++.


.. _building_source:

Building from source code
^^^^^^^^^^^^^^^^^^^^^^^^^

The source code is available on `GitHub <https://github.com/msesia/snpknock-python>`_.

Clone the repository and enter the source directory. Then, you can build the SNPknock package with::

  python setup.py build

To install, simply do::
   
  sudo python setup.py install


.. note::

    As with any python installation, this will install the modules
    in your system python *site-packages* directory (which is why you
    need *sudo*).  Many of us prefer to install development packages in a
    local directory so as to leave the system python alone.  This is
    mearly a preference, nothing will go wrong if you install using the
    *sudo* method. To install for the current user, use::

      python setup.py install --user

Installing via pip
^^^^^^^^^^^^^^^^^^

SNPknock can be easily installed through pip::
   
  sudo pip install SNPknock

.. note::

    As with any python installation, this will install the modules
    in your system python *site-packages* directory (which is why you
    need *sudo*).  Many of us prefer to install development packages in a
    local directory so as to leave the system python alone.  This is
    mearly a preference, nothing will go wrong if you install using the
    *sudo* method. To install for the current user, use::

      pip install SNPknock --user
    

Run tests
^^^^^^^^^^^^^^^^^^

To run the default tests, enter the main project directory type::
   
  pytest
