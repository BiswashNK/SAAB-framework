#!/bin/bash

# Check if dataset name is provided
if [ -z "$1" ]; then
    echo "Please provide a dataset name (e.g., asterisk, cwe119, cwe399)"
    echo "Usage: ./run_configurable.sh <dataset_name>"
    exit 1
fi

DATASET=$1
OUTPUT_DIR="runs/saab_bilstm_svm_${DATASET}_full"

# Verify dataset files exist
TRAIN_FILE="file/data/${DATASET}_ast_train.json"
TEST_FILE="file/data/${DATASET}_ast_test.json"
ADV_TEST_FILE="file/data/${DATASET}_ast_test_ADV.json"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

if [ ! -f "$ADV_TEST_FILE" ]; then
    echo "Warning: Adversarial test file not found: $ADV_TEST_FILE"
    echo "Will skip adversarial testing"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting pipeline for dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------"

echo "Step 1: Training on ${DATASET}_ast_train.json..."
python model/saab_model_run.py \
  --do_train \
  --train_data_file $TRAIN_FILE \
  --output_dir $OUTPUT_DIR \
  --block_size 512 \
  --train_batch_size 32 \
  --epochs 6 \
  --lr 1e-3 \
  --hidden 128 \
  --emb_dim 128 \
  --num_layers 3 \
  --dropout 0.1 \
  --use_shap \
  --shap_nsamples 100 \
  --shap_background 50

echo -e "\n----------------------------------------\n"

echo "Step 2: Running SHAP analysis..."
python model/analyze_shap.py --model_dir $OUTPUT_DIR

echo -e "\n----------------------------------------\n"

echo "Step 3: Testing on clean test set (${DATASET}_ast_test.json)..."
python model/saab_model_run.py \
  --do_test \
  --test_data_file $TEST_FILE \
  --output_dir $OUTPUT_DIR \
  --block_size 512 \
  --eval_batch_size 64

echo -e "\n----------------------------------------\n"

# Only run adversarial testing if file exists
if [ -f "$ADV_TEST_FILE" ]; then
    echo "Step 4: Testing on adversarial samples with SHAP defense..."
    echo "Step 4: Testing with standard model first..."
    # Run base model testing
    python model/saab_model_run.py \
      --do_test \
      --test_data_file $ADV_TEST_FILE \
      --clean_test_file $TEST_FILE \
      --output_dir $OUTPUT_DIR \
      --block_size 512 \
      --eval_batch_size 64

    # Save original predictions
    cp $OUTPUT_DIR/predictions.json $OUTPUT_DIR/predictions_original.json

    echo -e "\n----------------------------------------\n"

    echo "Step 5: Running SHAP defense analysis and correction..."
    # Run SHAP defense and capture detailed metrics
    python model/shap_defense.py \
      --model_dir $OUTPUT_DIR \
      --test_file $ADV_TEST_FILE \
      --output_file $OUTPUT_DIR/adv_defense_analysis.json
    
    # Copy defense results to predictions for evaluation
    cp $OUTPUT_DIR/adv_defense_analysis.json $OUTPUT_DIR/predictions.json

    # The defense has already updated predictions.json with corrected predictions
    cp $OUTPUT_DIR/predictions.json $OUTPUT_DIR/predictions_with_defense.json

    echo -e "\n----------------------------------------\n"

    echo -e "\nStep 6: Evaluating effectiveness of SHAP defense..."
    
    # Set up directories for before/after comparison
    mkdir -p $OUTPUT_DIR/before_defense $OUTPUT_DIR/after_defense
    
    # Copy model files needed for evaluation
    for DIR in before_defense after_defense; do
        mkdir -p $OUTPUT_DIR/$DIR
        cp $OUTPUT_DIR/encoder.pt $OUTPUT_DIR/$DIR/
        cp $OUTPUT_DIR/svm.pkl $OUTPUT_DIR/$DIR/
        cp $OUTPUT_DIR/vocab.txt $OUTPUT_DIR/$DIR/
    done
    
    echo -e "\nMetrics before SHAP defense (original model):"
    echo "----------------------------------------"
    python model/saab_model_run.py \
      --do_test \
      --test_data_file $ADV_TEST_FILE \
      --clean_test_file $TEST_FILE \
      --output_dir $OUTPUT_DIR/before_defense \
      --block_size 512 \
      --eval_batch_size 64 \
      --load_predictions $OUTPUT_DIR/predictions_original.json
    
    echo -e "\nMetrics after SHAP defense (corrected predictions):"
    echo "----------------------------------------"
    python model/saab_model_run.py \
      --do_test \
      --test_data_file $ADV_TEST_FILE \
      --clean_test_file $TEST_FILE \
      --output_dir $OUTPUT_DIR/after_defense \
      --block_size 512 \
      --eval_batch_size 64 \
      --load_predictions $OUTPUT_DIR/predictions.json
    
    # Show the impact of SHAP defense
    echo -e "\nSHAP Defense Impact:"
    echo "----------------------------------------"
    echo "Before defense:"
    cat $OUTPUT_DIR/before_defense/summary.json
    echo -e "\nAfter defense:"
    cat $OUTPUT_DIR/after_defense/summary.json
    
    echo "Final results:"
    echo "1. Original predictions (no defense): $OUTPUT_DIR/predictions_original.json"
    echo "2. Predictions after SHAP defense: $OUTPUT_DIR/predictions.json"
    echo "3. Defense analysis details: $OUTPUT_DIR/adv_defense_analysis.json"
fi

echo -e "\nAll results are saved in $OUTPUT_DIR"