import os
from distutils.core import setup

if os.path.exists("MANIFEST"):
    os.unlink("MANIFEST")

setup(name='hftools',
      version='0.4',
      packages=['hftools',
                'hftools._external',
                'hftools.constants', 'hftools.constants.tests',
                'hftools.core',
                'hftools.dataset', 'hftools.dataset.tests',
                'hftools.file_formats', 'hftools.file_formats.tests',
                'hftools.networks', 'hftools.networks.tests',
                'hftools.plotting', 
                'hftools.testing', 'hftools.tests',
                'hftools.tests',
                'hftools.testing.tests',
                ],
       license="BSD",
       author="The HFTools Development Team",
       platforms=['Linux','Mac OSX','Windows XP/Vista/7/8'],
       keywords=['microwave'],
       url="www.github.com/hftools",
       classifiers=['Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python',
                    'Programming Language :: Python :: 2',
                    'Programming Language :: Python :: 2.7',
#                    'Programming Language :: Python :: 3',
                        ],
       description="HFTools: Python tools for microwave engineering",
       long_description="""HFTools: Python tools for microwave engineering (long)""",
      )
