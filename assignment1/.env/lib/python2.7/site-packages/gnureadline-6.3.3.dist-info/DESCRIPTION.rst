Stand-alone GNU readline module
===============================

Some platforms, such as Mac OS X, do not ship with `GNU readline`_ installed.
The readline extension module in the standard library of Mac "system" Python
uses NetBSD's `editline`_ (libedit) library instead, which is a readline
replacement with a less restrictive software license.

As the alternatives to GNU readline do not have fully equivalent functionality,
it is useful to add proper readline support to these platforms. This module
achieves this by bundling the standard Python readline module with the GNU
readline source code, which is compiled and statically linked to it. The end
result is a package which is simple to install and requires no extra shared
libraries.

The module is called *gnureadline* so as not to clash with the readline module
in the standard library. This keeps polite installers such as `pip`_ happy and
is sufficient for shells such as `IPython`_. In order to use this module in
the standard Python shell it has to be installed with the more impolite
easy_install from `setuptools`_. **It is recommended that you use the latest
pip >= 1.4 together with setuptools >= 0.8 to install gnureadline.** This will
download a binary wheel from PyPI if available, thereby bypassing the need
for compilation and its slew of potential problems.

The module can be used with both Python 2.x and 3.x, and has been tested with
Python versions 2.6, 2.7, 3.2 and 3.3. The first three numbers of the module
version reflect the version of the underlying GNU readline library (major,
minor and patch level), while any additional fourth number distinguishes
different module updates based on the same readline library.

This module is usually unnecessary on Linux and other Unix systems with default
readline support. An exception is if you have a Python distribution that does
not include GNU readline due to licensing restrictions (such as ActiveState's
`ActivePython`_). If you are using Windows, which also ships without GNU 
readline, you might want to consider using the `pyreadline`_ module instead, 
which is a readline replacement written in pure Python that interacts with the
Windows clipboard. 

The latest development version is available from the `GitHub repository`_.

.. _GNU readline: http://www.gnu.org/software/readline/
.. _editline: http://www.thrysoee.dk/editline/
.. _pip: http://www.pip-installer.org/
.. _IPython: http://ipython.org/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _ActivePython: http://community.activestate.com/faq/why-doesnt-activepython-u
.. _pyreadline: http://pypi.python.org/pypi/pyreadline
.. _GitHub repository: http://github.com/ludwigschwardt/python-gnureadline


History
=======

6.3.3 (2014-04-08)
------------------

* Major rework of OS X build process (detect arches, no custom flags)
* #20, #22, #28: Various issues addressed by new streamlined build
* #28: Use $CC or cc to compile libreadline instead of default gcc
* #35: Workaround for clang from Xcode 5.1 and Mac OS X 10.9.2
* Uses Python 3.4 readline.c from hg 3.4 branch (89086:3110fb3095a2)
* Updated to build against readline 6.3 (patch-level 3)

6.2.5 (2014-02-19)
------------------

* Renamed module to *gnureadline* to improve installation with pip
* #23, #25-27, #29-33: Tweaks and package reworked to gnureadline
* Uses Python 2.x readline.c from hg 2.7 branch (89084:6b10943a5916)
* Uses Python 3.x readline.c from hg 3.3 branch (89085:6adac0d9b933)
* Updated to build against readline 6.2 (patch-level 5)

6.2.4.1 (2012-10-22)
--------------------

* #21: Fixed building on Python.org 3.3 / Mac OS 10.8

6.2.4 (2012-10-17)
------------------

* #15: Improved detection of compilers before Xcode 4.3
* Uses Python 3.x readline.c from v3.3.0 tag (changeset 73997)
* Updated to build against readline 6.2 (patch-level 4)

6.2.2 (2012-02-24)
------------------

* #14: Fixed compilation with Xcode 4.3 on Mac OS 10.7
* Updated to build against readline 6.2 (patch-level 2)

6.2.1 (2011-08-31)
------------------

* #10: Fixed '_emacs_meta_keymap' missing symbol on Mac OS 10.7
* #7: Fixed SDK version check to work with Mac OS 10.7 and later
* Uses Python 2.x readline.c from release27-maint branch (r87358)
* Uses Python 3.x readline.c from release32-maint branch (r88446)

6.2.0 (2011-06-02)
------------------

* #5: Removed '-arch ppc' on Mac OS 10.6, as Snow Leopard supports Intel only
* Updated to build against readline 6.2 (patch-level 1)

6.1.0 (2010-09-20)
------------------

* Changed version number to reflect readline version instead of Python version
* #4: Updated to build against readline 6.1 (patch-level 2)
* #2: Python 3 support
* Uses Python 2.x readline.c from release27-maint branch (r83672)
* Uses Python 3.x readline.c from r32a2 tag (r84541)
* Source code moved to GitHub
* Additional maintainer: Sridhar Ratnakumar

2.6.4 (2009-11-26)
------------------

* Added -fPIC to compiler flags to fix linking error on 64-bit Ubuntu
* Enabled all readline functionality specified in pyconfig.h macros
* Uses readline.c from Python svn trunk (r75725), which followed 2.6.4 release
* Patched readline.c to replace Py_XDECREF calls with the safer Py_CLEAR
* Fixed compilation error on Mac OS 10.4 with XCode older than version 2.4

2.6.1 (2009-11-18)
------------------

* Updated package to work with Mac OS 10.6 (Snow Leopard), which ships with 
  Python 2.6.1
* Uses readline.c from Python 2.6.1 release
* Backported "spurious trailing space" bugfix from Python svn trunk (see e.g. 
  https://bugs.launchpad.net/python/+bug/470824 for details on bug)
* Updated to build against readline 6.0 (patch-level 4)
* Now builds successfully on Linux (removed Mac-specific flags in this case),
  and still supports Mac OS 10.4 and 10.5

2.5.1 (2008-05-28)
------------------

* Updated package to work with Mac OS 10.5 (Leopard), which ships with Python 
  2.5.1
* Uses readline.c from Python 2.5.1 release
* Updated to build against readline 5.2 (patch-level 12)
* New maintainer: Ludwig Schwardt

2.4.2 (2005-12-26)
------------------

* Original package by Bob Ippolito, supporting Python 2.3 / 2.4 on Mac OS 10.3 
  (Panther) and 10.4 (Tiger)
* Builds against readline 5.1


