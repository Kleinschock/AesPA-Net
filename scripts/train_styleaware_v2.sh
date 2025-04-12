# --- Content for test_styleaware_v2.sh ---

# Optional: Define i if you want to run a single iteration outside a loop
# i=1 # Example: set i to 1

python main.py ^
    --type test ^
    --batch_size 1 ^
    --comment aepapa_run1 ^
    --content_dir ./content ^
    --style_dir ./style ^
    --num_workers 4 ^
    --test_result_dir ./test_output # Optional: define a specific output dir

    # Note: The original script had --test_iter $i.
    # If you run this directly without a loop, you might remove it
    # or set i manually as shown above. If it's part of a loop
    # in the .sh file, leave it as $i.
    # --test_iter $i ^