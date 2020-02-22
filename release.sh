#!/bin/bash

echo "Existing versions"
git tag -l | grep release

echo "Enter the next version"
read new_version

git tag -s release/${new_version}
git push origin master release/${new_version}
rm -rf dist
python setup.py sdist
twine upload dist/*
