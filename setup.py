from setuptools import setup

setup(
    name='SpikePy',

    version='0.3.0',

    url='https://gitlab.com/SpikeLabPublic/SpikePy',

    description='Python package with helper functions for H2o, Lime and others',

    py_modules=['data_utils', 'h2o_bigquery', 'h2o_interface', 'production']
)