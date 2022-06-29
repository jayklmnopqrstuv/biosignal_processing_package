from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rndSignal",
    author=author,
    author_email=author_email,
    description="For standardizing signal preprocessing at OptumLabs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.savvysherpa.com/SavvysherpaResearch/rndSignal",
    packages=['rndSignal', 'rndSignal/extract_features', 
              'rndSignal/plotting', 'rndSignal/preprocessing', 
              'rndSignal/savvyppg', 'rndSignal/sensorlib', 
              'rndSignal/signal_quality'
             ],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    package_data={'rndSignal': 
                  ['rndSignal/savvyppg/models/randomforest_pca.pkl', 
                   'rndSignal/signal_quality/deepbeat.h5']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        "numpy",
        "pandas",
        "bokeh",
        "biosignalsnotebooks",
        "neurokit2",
        "python-magic-bin==0.4.14",
        "pywavelets",
        "scikit-image",
        "scikit-learn",
        "statsmodels",
        "tensorflow"
    ]
)
