rm -r examples

# agreement
python generate.py -d examples/agreement/oneshape -U -t agreement -n oneshape -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/oneshape_simple -U -t agreement -n oneshape_simple -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/multishape -U -t agreement -n multishape -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/multishape_simple -U -t agreement -n multishape_simple -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/spatial -U -t agreement -n spatial -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/spatial_simple -U -t agreement -n spatial_simple -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/relational -U -t agreement -n relational -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/counting -U -t agreement -n counting -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/counting_simple -U -t agreement -n counting_simple -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/quantification -U -t agreement -n quantification -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/quantification_simple -U -t agreement -n quantification_simple -m train -i 100 -p 0.1 -M
python generate.py -d examples/agreement/combination -U -t agreement -n combination -m train -i 100 -p 0.1 -M

# classification
python generate.py -d examples/classification/oneshape -U -t classification -n oneshape -m train -i 100 -p 0.1 -M
python generate.py -d examples/classification/multishape -U -t classification -n multishape -m train -i 100 -p 0.1 -M

# readme data
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -f "(5,5,1,1)" -i 128 -M -S

# load and evaluate
python evaluate.py -t agreement -n multishape -c "load(examples/readme)" -m cnn_bow -i 10 -T
