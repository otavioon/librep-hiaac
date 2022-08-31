.. _installation:

==========================
Installation Guide
==========================

Librep can be installed in multiple ways, including via docker and pip

Pip Installation
---------------------

The package can be installed directly from pip using

.. code-block:: bash

  pip install git+https://github.com/otavioon/librep-hiaac.git


Librep can be used as a normal python package, as below:

.. code-block:: python

  from librep.base.transform import Transform

  class DummyTransform(Transform):
      # X is a single sample. Returns X
      def transform(self, X):
          return X

  my_transform = DummyTransform()
  result = my_transform.transform([1,2,3])
  ...

See the API for more details.


Editable Jupyter-lab Docker Container Environment
----------------------------------------------------

A built-in jupyter-lab environment with docker is also available.
The user must first download the package from github:

.. code-block:: bash

  git clone --recurse-submodules https://github.com/otavioon/librep-hiaac.git

This will download librep package and additional data from data librep-hiaac-data.git.
The data includes some KuHar and MotionSense views.

To use the full jupyter-lab environment, we first need to generate some SSL keys.
An auxiliary script is available to perform this operation.

.. code-block:: bash

  ./generate_ssl_keys.sh

Several environment variables can be configured simple editing the `vars.sh` file.
If no edit was made, it will use the defaults.

The user can run the script to generate docker image, using the script:

.. code-block:: bash

  ./build_docker.sh

.. note::

  This is a CPU image and not an GPU image


After the docker build, the user can start a jupyter-lab server using the helper script:

.. code-block:: bash

  ./start_jupyter_lab_server.sh

It will launch a jupyter script listing on port defined in the `vars.sh` file.
The container will map the git's root directory inside.
It will be ready to use.
