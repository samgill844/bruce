#python setup.py build_ext --inplace
python setup.py clean --all
python setup.py install
python setup.py clean --all
rm -r dist build src/bruce/__pycache__