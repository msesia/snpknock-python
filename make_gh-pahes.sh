#!/bin/sh

GH_PAGES_SOURCES="docs/source docs/Makefile"

git checkout gh-pages
rm -rf build _sources _static _modules
git checkout master $GH_PAGES_SOURCES
git reset HEAD
cd docs
make html
cd ..
mv -fv docs/build/html/* ./
rm -rf $GH_PAGES_SOURCES docs/build

git add -A
git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
git push origin gh-pages
git checkout master
