# agreement
rm -r examples/agreement

# existential
python -O generate.py -d examples/agreement/existential-oneshape -U -t agreement -n existential \
    -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --entity-counts [1] --train-entity-counts [1] --validation-entity-counts [1] \
    --test-entity-counts [1] --captions-per-instance 5

python -O generate.py -d examples/agreement/existential-colfree -U -t agreement -n existential \
    -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --captions-per-instance 5

python -O generate.py -d examples/agreement/existential -U -t agreement -n existential -i 100 -M -H \
    -N --v1 --config-values --world-size 100 --captions-per-instance 5

# relational
python -O generate.py -d examples/agreement/relational-spatial_twoshapes -U -t agreement \
    -n relational -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --entity-counts [2] --train-entity-counts [2] --validation-entity-counts [2] \
    --test-entity-counts [2] --relations '[["x-rel", "y-rel"], null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/relational-spatial -U -t agreement -n relational -i 100 \
    -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --relations '[["x-rel", "y-rel"], null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/relational-comparative -U -t agreement -n relational \
     -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
     --relations '[["size-rel", "shade-rel"], null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/relational -U -t agreement -n relational -i 100 -M -H -N \
    --v1 --config-values --world-size 100 --captions-per-instance 5

# quantification
python -O generate.py -d examples/agreement/quantification-count_equal -U -t agreement \
    -n quantification -i 100 -M -H -N --v1 --config-values --world-size 100 \
    --collision_tolerance 0.0 --quantifiers '[["count"], ["eq"], "+"]' --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification-count_nonsubtr -U -t agreement \
    -n quantification -i 100 -M -H -N --v1 --config-values --world-size 100 \
    --collision_tolerance 0.0 --quantifiers '[["count"], null, "+"]' --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification-count -U -t agreement -n quantification \
    -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --quantifiers '[["count"], null, null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification-ratio_equal -U -t agreement \
    -n quantification -i 100 -M -H -N --v1 --config-values --world-size 100 \
    --collision_tolerance 0.0 --quantifiers '[["ratio"], ["eq"], null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification-ratio -U -t agreement -n quantification \
    -i 100 -M -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --quantifiers '[["ratio"], null, null]' --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification -U -t agreement -n quantification -i 100 \
    -M -H -N --v1 --config-values --world-size 100 --captions-per-instance 5

python -O generate.py -d examples/agreement/quantification_complex -U -t agreement \
    -n quantification_complex -i 100 -M -H -N --v1 --config-values --world-size 100 \
    --captions-per-instance 5

# logical
python -O generate.py -d examples/agreement/logical-existential -U -t agreement -n logical -i 100 -M \
    -H -N --v1 --config-values --world-size 100 --collision_tolerance 0.0 \
    --captions '["existential"]' --captions-per-instance 5

python -O generate.py -d examples/agreement/logical -U -t agreement -n logical -i 100 -M -H -N \
    --v1 --config-values --world-size 100 --captions-per-instance 5

# classification
rm -r examples/classification

python -O generate.py -d examples/classification/shape-single -U -t classification -n shape -i 100 \
    -M -H -N --v1 --config-values --world-size 100 --multi-class false

python -O generate.py -d examples/classification/shape-multi -U -t classification -n shape -i 100 -M \
    -H -N --v1

python -O generate.py -d examples/classification/shape-count -U -t classification -n shape -i 100 -M \
    -H -N --v1 --config-values --world-size 100 --count-class true

# readme data
rm -r examples/readme

python -O generate.py -d examples/readme -a tar:bzip2 -t agreement -n existential -v readme -s 3,2,1 \
    -i 100 -M -T --v1

python -O generate.py -d examples/readme -a tar:bzip2 -t agreement -n relational -v readme -s 3,2,1 \
    -i 100 -M --v1
