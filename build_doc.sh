# source activate .docs-venv
rm -rf docs
sphinx-build docs_source/source/ docs
echo " " > docs/.nojekyll
# deactivate
